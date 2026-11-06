"""Prepara o C4 (en) tokenizado com o BPE do GPT-2 em arquivos binários uint16.

Saída (em data/c4/):
  train.bin, validation.bin               tokens concatenados, documentos separados por EOT
  train_meta.json, validation_meta.json   configuração, contagens e SHA-256 do .bin
  *.state.json                            progresso, permite retomar se o streaming cair

Referências: C4 em Raffel et al. (JMLR 2020); versão allenai/c4 documentada em
Dodge et al. (EMNLP 2021). Documentos concatenados e separados por
<|endoftext|>, como no GPT-2/GPT-3 (Radford et al. 2019; Brown et al. 2020).

Reprodutibilidade:
- O dataset é fixado num commit (revision) do Hugging Face Hub, resolvido na
  primeira execução e gravado no estado e nos metadados.
- O estado guarda a configuração completa; uma retomada com configuração
  diferente é recusada.
- Na retomada, os últimos documentos gravados são recomparados com o stream
  antes de continuar, o que confirma que o streaming reproduz a mesma ordem.
- Ao final, o SHA-256 do .bin vai para os metadados, para que qualquer
  membro confira se tem exatamente o mesmo arquivo.

Uso:
  python prepare_c4.py                 # prepara train e validation (retoma se houver progresso)
  python prepare_c4.py --no_resume     # recomeça do zero
  python prepare_c4.py --verify_only   # só verifica os .bin existentes (inclui SHA-256)
"""

import os
import sys
import argparse
import hashlib
import itertools
from collections import deque

import numpy as np

from common import TOKEN_DTYPE, write_json, read_json

DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en"
TOKENIZER_NAME = "gpt2"

SPLIT_LIMITS = {
    "train": 3_000_000_000,
    "validation": 100_000_000,
}

SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 10_000
DOC_BATCH = 1024             
FLUSH_TOKENS = 10_000_000 # tokens gravados entre pontos de retomada
RESUME_CHECK_DOCS = 100 # documentos recomparados com o stream na retomada

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "c4")


class ResumeError(RuntimeError):
    pass


def get_encoder():
    import tiktoken
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    assert enc.n_vocab <= np.iinfo(TOKEN_DTYPE).max + 1, "vocabulário não cabe em uint16"
    return enc


def resolve_revision():
    """Commit atual do dataset no Hub, para fixar a versão usada."""
    from huggingface_hub import HfApi
    return HfApi().dataset_info(DATASET_NAME).sha


def stream_texts(split, revision, shuffle=True):
    """Fluxo determinístico de textos: mesmo commit + mesma semente -> mesma ordem."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split,
                      streaming=True, revision=revision)
    if shuffle:
        ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    for example in ds:
        yield example["text"]


def split_config(enc, split, max_tokens, revision):
    """Tudo o que determina o conteúdo do .bin. Deve ser idêntico na retomada."""
    return {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "revision": revision,
        "split": split,
        "tokenizer": TOKENIZER_NAME,
        "n_vocab": enc.n_vocab,
        "eot_token": enc.eot_token,
        "dtype": np.dtype(TOKEN_DTYPE).name,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_buffer": SHUFFLE_BUFFER,
        "max_tokens": max_tokens,
    }


def library_versions():
    versions = {}
    for name in ("datasets", "tiktoken", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def iter_document_tokens(enc, texts, num_threads):
    """Tokeniza documentos em lotes (multithread) e anexa EOT a cada um."""
    texts = iter(texts)
    while True:
        batch = list(itertools.islice(texts, DOC_BATCH))
        if not batch:
            return
        for tokens in enc.encode_ordinary_batch(batch, num_threads=num_threads):
            tokens.append(enc.eot_token)
            yield tokens


def file_sha256(path, chunk_bytes=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def sanity_check(enc, texts, num_threads):
    texts = list(texts)
    eot = enc.eot_token
    batch_tokens = enc.encode_ordinary_batch(texts, num_threads=num_threads)
    errors = []

    for i, (text, tokens) in enumerate(zip(texts, batch_tokens)):
        if tokens != enc.encode_ordinary(text):
            errors.append(f"doc {i}: tokenização em lote difere da sequencial")
        if eot in tokens:
            errors.append(f"doc {i}: EOT gerado dentro do documento")
        if enc.decode(tokens) != text:
            errors.append(f"doc {i}: decode(encode(texto)) != texto")

    packed = [t for doc in iter_document_tokens(enc, texts, num_threads) for t in doc]
    packed = np.asarray(packed, dtype=np.int64)
    boundaries = np.flatnonzero(packed == eot)
    if len(boundaries) != len(texts):
        errors.append(f"EOTs: {len(boundaries)} para {len(texts)} documentos")
    else:
        start = 0
        for i, end in enumerate(boundaries):
            if enc.decode(packed[start:end].tolist()) != texts[i]:
                errors.append(f"doc {i}: não recuperado após empacotamento")
                break
            start = end + 1

    if errors:
        print("FALHA no teste de sanidade:")
        for e in errors[:20]:
            print("  -", e)
        sys.exit(1)

    print(f"Teste de sanidade OK: {len(texts):,} documentos, {len(packed):,} tokens "
          f"(lote = sequencial, ida-e-volta exata, sem EOT interno, empacotamento reversível).")

def last_documents_on_disk(bin_path, num_tokens, k):
    """Tokens dos últimos k documentos gravados (cada um sem o EOT final)."""
    data = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")[:num_tokens]
    eot = None
    window = 1_000_000
    while True:
        tail = np.asarray(data[max(0, num_tokens - window):], dtype=np.int64)
        if eot is None:
            eot = int(tail[-1])  # todo arquivo gravado termina em EOT
        ends = np.flatnonzero(tail == eot)
        if len(ends) > k or window >= num_tokens:
            break
        window *= 4
    starts = np.concatenate(([0], ends[:-1] + 1))
    # Se a janela não começa no início do arquivo, o primeiro segmento pode
    # estar cortado: só os segmentos depois do primeiro EOT são confiáveis.
    usable = list(zip(starts, ends))
    if window < num_tokens:
        usable = usable[1:]
    return [tail[s:e].tolist() for s, e in usable[-k:]]


def resume_iterator(enc, texts_factory, docs_done, bin_path, tokens_done, num_threads):
    """Avança o stream até docs_done, recomparando os últimos documentos com o
    que está no disco. Retorna o iterador posicionado no próximo documento."""
    k = min(RESUME_CHECK_DOCS, docs_done)
    texts = texts_factory()
    if k == 0:
        return texts
    for _ in itertools.islice(texts, docs_done - k):
        pass
    expected_texts = [next(texts) for _ in range(k)]
    expected = enc.encode_ordinary_batch(expected_texts, num_threads=num_threads)
    on_disk = last_documents_on_disk(bin_path, tokens_done, k)

    if len(on_disk) != k or any(a != b for a, b in zip(expected, on_disk)):
        raise ResumeError(
            "Os últimos documentos gravados não coincidem com o stream: a ordem do "
            "streaming mudou desde a execução anterior. Rode com --no_resume."
        )
    print(f"  retomada verificada: últimos {k} documentos idênticos ao stream.")
    return texts

def prepare_split(enc, split, max_tokens, texts_factory, num_threads, revision, resume=True):
    bin_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
    state_path = os.path.join(OUTPUT_DIR, f"{split}.state.json")
    meta_path = os.path.join(OUTPUT_DIR, f"{split}_meta.json")
    itemsize = np.dtype(TOKEN_DTYPE).itemsize
    config = split_config(enc, split, max_tokens, revision)

    docs, tokens = 0, 0
    mode = "wb"
    if resume and os.path.exists(state_path) and os.path.exists(bin_path):
        state = read_json(state_path)
        saved = state.get("config")
        if saved != config:
            diffs = sorted(k for k in set(config) | set(saved or {})
                           if (saved or {}).get(k) != config.get(k))
            raise ResumeError(
                f"{split}: configuração diferente da execução anterior ({diffs}). "
                "Rode com --no_resume para recomeçar."
            )
        if state.get("complete"):
            print(f"{split}: já concluído ({state['tokens']:,} tokens). Pulando.")
            return read_json(meta_path)
        if os.path.getsize(bin_path) < state["tokens"] * itemsize:
            raise ResumeError(f"{split}: .bin menor que o registrado no estado.")
        docs, tokens = state["docs"], state["tokens"]
        mode = "r+b"
        print(f"{split}: retomando de {docs:,} documentos / {tokens:,} tokens.")
        saved_versions = state.get("library_versions", {})
        if saved_versions and saved_versions != library_versions():
            print(f"  Aviso: versões de bibliotecas mudaram ({saved_versions} -> "
                  f"{library_versions()}); a recomparação abaixo confirma a ordem.")

    buffer, buffered = [], 0
    stop_reason = "fim_do_stream"

    def state_record(complete):
        return {"docs": docs, "tokens": tokens, "complete": complete,
                "config": config, "library_versions": library_versions()}

    with open(bin_path, mode) as f:
        if mode == "r+b":
            # Descarta qualquer escrita parcial posterior ao último ponto salvo.
            f.truncate(tokens * itemsize)
            f.seek(0, os.SEEK_END)
            texts = resume_iterator(enc, texts_factory, docs, bin_path, tokens, num_threads)
        else:
            texts = texts_factory()
            write_json(state_path, state_record(False))

        def flush():
            nonlocal buffer, buffered
            if buffer:
                np.concatenate(buffer).astype(TOKEN_DTYPE).tofile(f)
                f.flush()
                os.fsync(f.fileno())
                buffer, buffered = [], 0
            # O estado só é atualizado depois que os tokens estão no disco.
            write_json(state_path, state_record(False))

        for doc_tokens in iter_document_tokens(enc, texts, num_threads):
            # Corte na fronteira de documento: nenhum documento é truncado.
            if tokens + len(doc_tokens) > max_tokens:
                stop_reason = "limite_de_tokens"
                break
            buffer.append(np.asarray(doc_tokens, dtype=TOKEN_DTYPE))
            buffered += len(doc_tokens)
            tokens += len(doc_tokens)
            docs += 1
            if buffered >= FLUSH_TOKENS:
                flush()
                print(f"  {split}: {tokens:,} tokens / {docs:,} documentos", flush=True)

        flush()

    if stop_reason == "fim_do_stream" and tokens < max_tokens:
        print(f"Aviso: o stream de '{split}' terminou antes do limite ({tokens:,} < {max_tokens:,}).")

    meta = dict(config)
    meta.update({
        "num_tokens": tokens,
        "num_documents": docs,
        "mean_tokens_per_document": tokens / max(docs, 1),
        "tokens_below_limit": max_tokens - tokens,
        "stop_reason": stop_reason,
        "library_versions": library_versions(),
    })
    write_json(meta_path, meta)
    write_json(state_path, state_record(True))
    print(f"{split}: {tokens:,} tokens, {docs:,} documentos ({stop_reason}; "
          f"{max_tokens - tokens:,} abaixo do limite).")
    return meta


def verify_split(enc, split, texts_factory, num_docs=200):
    """Confere o arquivo gravado:
    - tamanho igual aos metadados e término em EOT;
    - número de EOTs igual ao número de documentos (estrutura do arquivo todo);
    - reconstrução exata dos primeiros documentos a partir do stream;
    - SHA-256 do arquivo inteiro, gravado nos metadados (ou conferido, se já existir).
    """
    bin_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
    meta_path = os.path.join(OUTPUT_DIR, f"{split}_meta.json")
    meta = read_json(meta_path)
    data = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")
    eot = enc.eot_token

    assert len(data) == meta["num_tokens"], "tamanho do .bin difere dos metadados"
    assert data[-1] == eot, "o arquivo não termina em EOT"

    step = 100_000_000
    eot_count = sum(
        int(np.count_nonzero(data[i:i + step] == eot))
        for i in range(0, len(data), step)
    )
    assert eot_count == meta["num_documents"], (
        f"EOTs ({eot_count:,}) != documentos ({meta['num_documents']:,})"
    )

    head = np.asarray(data[:min(len(data), 5_000_000)], dtype=np.int64)
    ends = np.flatnonzero(head == eot)[:num_docs]
    reference = list(itertools.islice(texts_factory(), len(ends)))
    start = 0
    for i, end in enumerate(ends):
        if enc.decode(head[start:end].tolist()) != reference[i]:
            raise AssertionError(f"{split}: documento {i} difere do stream")
        start = end + 1

    digest = file_sha256(bin_path)
    if meta.get("sha256") and meta["sha256"] != digest:
        raise AssertionError(f"{split}: SHA-256 difere do registrado nos metadados")
    meta["sha256"] = digest
    write_json(meta_path, meta)

    print(f"{split}: verificação OK ({eot_count:,} documentos, {len(ends)} primeiros "
          f"conferidos com o stream, SHA-256 {digest[:16]}…).")


def main():
    parser = argparse.ArgumentParser(description="Preparação do C4 (en) com tokenizador GPT-2")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_LIMITS), choices=list(SPLIT_LIMITS))
    parser.add_argument("--num_threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_sanity", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--verify_docs", type=int, default=200)
    parser.add_argument("--revision", default=None,
                        help="commit do dataset no Hub (padrão: o registrado no estado, "
                             "ou o atual na primeira execução)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enc = get_encoder()
    print(f"Tokenizador: {TOKENIZER_NAME} | vocabulário: {enc.n_vocab:,} | EOT: {enc.eot_token}")

    # Mesma revisão para todos os splits: a já registrada, a informada ou a atual.
    revision = args.revision
    if revision is None and not args.no_resume:
        for split in args.splits:
            for name in (f"{split}.state.json", f"{split}_meta.json"):
                path = os.path.join(OUTPUT_DIR, name)
                if os.path.exists(path):
                    saved = read_json(path)
                    revision = (saved.get("config") or saved).get("revision")
                    if revision:
                        break
            if revision:
                break
    if revision is None:
        revision = resolve_revision()
    print(f"Dataset: {DATASET_NAME}/{DATASET_CONFIG} @ {revision}")
    print(f"Threads de tokenização: {args.num_threads}")

    if not args.skip_sanity and not args.verify_only:
        print("Teste de sanidade (1.000 documentos de treino)...")
        sanity_check(enc, itertools.islice(stream_texts("train", revision, shuffle=False), 1000),
                     args.num_threads)

    for split in args.splits:
        factory = lambda s=split: stream_texts(s, revision, shuffle=True)
        if not args.verify_only:
            print(f"\nPreparando {split}...")
            prepare_split(enc, split, SPLIT_LIMITS[split], factory,
                          args.num_threads, revision, resume=not args.no_resume)
        verify_split(enc, split, factory, num_docs=args.verify_docs)

    print("\nPreparação do C4 concluída.")


if __name__ == "__main__":
    main()
