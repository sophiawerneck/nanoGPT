"""Treinamento das variantes de bloco (seq, inv, par) do GPT (~124M) no C4.

Fluxo de Funcionamento:
- Inicialização e configuração: Valida hiperparâmetros, configura o ambiente 
  distribuído (DDP) se exigido, fixa as sementes (reprodutibilidade) e carrega 
  as identidades (hashes/tamanhos) dos arquivos de dados gerados na preparação.
- Amostragem de dados: Estabelece o `TrainSampler`, de forma a garantir uma amostragem 
  determinística sem reposição. Separa subconjuntos estáticos de treino e validação 
  para monitoramento isolado e avaliações imparciais.
- Modelo e retomada: Constrói a arquitetura GPT do zero (com compilação opcional 
  via torch.compile) ou restaura o estado exato de um checkpoint (pesos, otimizador, 
  RNG states e passo iterativo).
- Loop de treino: Computa o decaimento cosseno da taxa de aprendizado, propaga 
  os micro-lotes acumulando gradientes (mixed precision via GradScaler), aplica 
  clipping e atualiza os parâmetros.
- Avaliação e registro (Logging): Periodicamente avalia as perdas de validação e 
  treino sem impactar os gradientes, salva checkpoints e exporta estatísticas vitais 
  para um CSV de métricas em tempo real (throughput, step time, MFU).
- Finalização: Concluído o limite de orçamento (max_iters), gera e grava um 
  summary.json consolidando o custo computacional e os resultados finais.

Uma GPU:
  python train.py --block_type seq --seed 1337
DDP em 4 GPUs de um nó:
  torchrun --standalone --nproc_per_node=4 train.py --block_type seq --seed 1337
Retomada:
  python train.py --block_type seq --seed 1337 --init_from resume

Escolhas metodológicas:
- Hiperparâmetros do GPT-3 Small (Brown et al., NeurIPS 2020): ~0,5M tokens
  por passo, lr 6e-4, AdamW (Loshchilov & Hutter, ICLR 2019) com
  beta=(0,9; 0,95), weight decay 0,1, clipping 1,0, decaimento cosseno até 10%.
- Orçamento de ~3B tokens para ~124M parâmetros (~24 tokens/parâmetro), perto
  do ótimo computacional de ~20 de Hoffmann et al. (NeurIPS 2022).
- Amostragem sem reposição: o train.bin é dividido em janelas não sobrepostas
  de block_size tokens, e cada janela é sorteada uma única vez por época,
  numa permutação definida pela semente. Como as janelas não se sobrepõem,
  cada token aparece como entrada no máximo uma vez por época; os tokens
  restantes no fim do arquivo (menos de uma janela) não são usados. O
  orçamento cobre menos de uma época, então não há repetição de dados, o que
  evita o efeito da repetição sobre a perda (Muennighoff et al., NeurIPS 2023).
  Mesma semente = mesma ordem de dados em todas as variantes (comparação
  pareada).
- Monitoramento em subconjuntos fixos de treino e validação; as janelas de
  validação a partir de val_monitor_windows ficam reservadas para a avaliação
  final.
- Tempo de passo medido com sincronização da GPU, excluindo a avaliação e as
  primeiras iterações de cada processo (compilação e aquecimento).

Reprodutibilidade: a semente, a configuração, as versões e a identidade dos
dados são registradas, e a retomada reproduz a ordem dos dados exatamente.
Resultados numéricos bit a bit idênticos não são garantidos entre GPUs,
plataformas ou versões do PyTorch (ver "Reproducibility" na documentação do
PyTorch).
"""

from __future__ import annotations

import os
import time
import math
import random
import csv
import argparse
from datetime import timedelta
from typing import Any, Callable
import pickle
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from common import (
    BASE_DIR, count_windows, load_windows, default_dtype, autocast_ctx, sync,
    strip_compile_prefix, count_params, environment_info, write_json, read_json,
    file_sha256,
)

DATASET_PRESETS: dict[str, dict[str, Any]] = {
    'c4': {},
    'shakespeare_char': dict(
        n_layer=6, n_head=6, n_embd=384, dropout=0.2,
        block_size=256, batch_size=64, grad_accum=1,
        learning_rate=1e-3, min_lr=1e-4, beta2=0.99,
        max_iters=5000, lr_decay_iters=5000, warmup_iters=100,
        eval_interval=250, log_interval=10,
        eval_iters=4,
        allow_multi_epoch=True,
    ),
}
DATASET_HINTS = {
    'c4': 'python prepare_c4.py',
    'shakespeare_char': 'python data/shakespeare_char/prepare.py',
}

GPT2_PADDED_VOCAB = 50304
SMALL_FILE_BYTES = 512 * 1024 * 1024

MONITOR_SEED = 12345

RESUME_LOCKED_KEYS = [
    'batch_size', 'block_size', 'grad_accum', 'learning_rate', 'min_lr',
    'max_iters', 'lr_decay_iters', 'warmup_iters', 'weight_decay', 'beta1',
    'beta2', 'grad_clip', 'decay_lr', 'seed', 'eval_iters', 'eval_interval',
    'block_type', 'n_layer', 'n_head', 'n_embd', 'bias', 'dropout',
    'dtype',
    'dataset', 'data_fingerprint',
]

METRIC_COLUMNS = [
    'iter',
    'train_loss',
    'val_loss',
    'train_ppl',
    'val_ppl',
    'lr',
    'step_time_ms',
    'tokens_per_s',
    'train_time_s',
    'eval_time_s',
]

class TrainSampler:
    """Ordem sem reposição sobre as janelas de treino.

    A amostra global de índice s pertence à época s // n e ocupa a posição
    s % n de uma permutação gerada com (seed, época). O lote de qualquer
    iteração pode ser recalculado a partir de iter_num, o que torna a
    retomada exata sem guardar o estado do carregador.
    """

    def __init__(self, num_windows: int, seed: int) -> None:
        if num_windows <= 0:
            raise ValueError("num_windows deve ser positivo")
        self.n = num_windows
        self.seed = seed
        self._epoch = -1
        self._perm: np.ndarray | None = None

    def _perm_for(self, epoch: int) -> np.ndarray:
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = rng.permutation(self.n)
            self._epoch = epoch
        return self._perm

    def ids(self, start: int, count: int) -> np.ndarray:
        """Índices das janelas das amostras globais start, ..., start + count - 1."""
        out = np.empty(count, dtype=np.int64)
        for j in range(count):
            epoch, pos = divmod(start + j, self.n)
            out[j] = self._perm_for(epoch)[pos]
        return out


def global_micro_batch(it: int, micro: int, rank: int, micro_per_rank: int, world_size: int) -> int:
    """Índice global do micro-lote (iteração, micro-passo, rank).

    Cada iteração consome micro_per_rank * world_size micro-lotes consecutivos,
    então o conjunto de dados de uma iteração não depende do número de GPUs.
    """
    return it * micro_per_rank * world_size + micro * world_size + rank

def data_fingerprint(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Identidade dos arquivos de dados: tamanho em bytes e SHA-256.

    Para o C4, o SHA-256 vem dos metadados de prepare_c4.py (calcular o hash
    de 6 GB a cada execução seria lento). Para datasets sem esses metadados,
    como o Tiny Shakespeare, o hash é calculado na hora.
    """
    fp = {}
    for path in paths:
        name = os.path.basename(path)
        meta_path = os.path.join(os.path.dirname(path), name.replace('.bin', '_meta.json'))
        sha = read_json(meta_path).get('sha256') if os.path.exists(meta_path) else None
        if sha is None and os.path.getsize(path) <= SMALL_FILE_BYTES:
            sha = file_sha256(path)
        fp[name] = {'bytes': os.path.getsize(path), 'sha256': sha}
    return fp

def find_val_file(data_dir: str) -> str:
    for name in ('validation.bin', 'val.bin'):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(data_dir, 'validation.bin')


def dataset_vocab_size(data_dir: str) -> int:
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            return int(pickle.load(f)['vocab_size'])
    return GPT2_PADDED_VOCAB

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, ctx, monitor: dict[str, tuple[str, np.ndarray]],
                  batch_size: int, block_size: int, device: str,
                  device_type: str) -> dict[str, float]:
    """Loss média em subconjuntos fixos (sem sorteio) de treino e validação.

    Coloca o modelo em modo de avaliação e o devolve ao modo de treino.
    """
    out = {}
    model.eval()
    for split, (path, ids) in monitor.items():
        losses = []
        for k in range(0, len(ids), batch_size):
            X, Y = load_windows(path, ids[k:k + batch_size], block_size, device, device_type)
            with ctx:
                _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out
  
def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float,
           min_lr: float) -> float:
    """Aquecimento linear seguido de decaimento cosseno até min_lr."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_rng_state() -> dict[str, Any]:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def collect_rng_states(ddp: bool) -> list[dict[str, Any]]:
    """Estados aleatórios de todos os ranks (relevantes se dropout > 0).

    Em DDP é uma op coletiva: todos os ranks precisam chamá-la.
    """
    local = get_rng_state()
    if ddp:
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
        return states
    return [local]


def save_checkpoint(path: str, checkpoint: dict[str, Any]) -> None:
    tmp = path + '.tmp'
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def new_timing() -> dict[str, float]:
    return {
        'train_time_s': 0.0,
        'warmup_time_s': 0.0,
        'steady_time_s': 0.0,
        'steady_sq_s': 0.0,
        'steady_iters': 0,
        'eval_time_s': 0.0,
    }


def prepare_metrics_file(metrics_file: str, init_from: str, resume_iter: int, reevaluates_resume_iter: bool) -> None:
    """Cria o CSV de métricas (do zero) ou o prepara para a retomada.

    Na retomada mantém as linhas anteriores; a linha da própria iteração de
    retomada só é descartada se ela for avaliada de novo.
    """
    previous_rows = []
    limit = resume_iter if reevaluates_resume_iter else resume_iter + 1
    if init_from == 'resume' and os.path.exists(metrics_file):
        with open(metrics_file, 'r', newline='') as f:
            for row in csv.DictReader(f):
                try:
                    if int(row['iter']) < limit:
                        previous_rows.append(row)
                except (ValueError, KeyError):
                    continue
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(previous_rows)


def evaluate_and_save_checkpoint(
    iter_num: int, raw_model, optimizer, scaler, ctx, cfg: Config, rt: Runtime,
    timing: dict[str, float], lr: float, interval_time: float, interval_iters: int,
    tokens_per_iter: int
) -> dict[str, float] | None:
    if rt.ddp:
        dist.barrier()
    rng_states = collect_rng_states(rt.ddp)
    losses = None

    if rt.master:
        sync(rt.device_type)
        t_eval = time.perf_counter()
        losses = estimate_loss(raw_model, ctx, rt.monitor, cfg.batch_size,
                               cfg.block_size, rt.device, rt.device_type)
        sync(rt.device_type)
        timing['eval_time_s'] += time.perf_counter() - t_eval

        if interval_iters > 0:
            step_s = interval_time / interval_iters
            step_ms, tok_s = step_s * 1000, tokens_per_iter / step_s
        else:
            step_ms, tok_s = float('nan'), float('nan')

        print(f"step {iter_num}: train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | val ppl {math.exp(losses['val']):.2f} | "
              f"passo {step_ms:.1f} ms | {tok_s:,.0f} tok/s")

        if not cfg.eval_only:
            with open(rt.metrics_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    iter_num,
                    f"{losses['train']:.4f}",
                    f"{losses['val']:.4f}",
                    f"{math.exp(losses['train']):.2f}",
                    f"{math.exp(losses['val']):.2f}",
                    f"{lr:.6e}",
                    f"{step_ms:.2f}",
                    f"{tok_s:.0f}",
                    f"{timing['train_time_s']:.2f}",
                    f"{timing['eval_time_s']:.2f}",
                ])

        if iter_num > 0 and not cfg.eval_only:
            save_checkpoint(rt.checkpoint_path, {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'model_args': cfg.model_args,
                'config': cfg.as_dict(),
                'iter_num': iter_num,
                'val_loss': losses['val'],
                'timing': dict(timing),
                'rng_states': rng_states,
                'world_size': rt.world_size,
            })
            print(f"Checkpoint salvo em {rt.checkpoint_path}")

    if rt.ddp:
        dist.barrier()
        
    return losses


def perform_training_step(
    iter_num: int, model, optimizer, scaler, ctx, cfg: Config, rt: Runtime,
    train_batch_fn: Callable, X: torch.Tensor, Y: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    micro_per_rank = cfg.grad_accum_per_rank
    loss_accum = torch.zeros((), device=rt.device)
    
    for micro in range(micro_per_rank):
        if rt.ddp:
            model.require_backward_grad_sync = (micro == micro_per_rank - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / micro_per_rank
        loss_accum += loss.detach().float()
        
        if micro + 1 < micro_per_rank:
            X, Y = train_batch_fn(iter_num, micro + 1)
        elif iter_num + 1 < cfg.max_iters:
            X, Y = train_batch_fn(iter_num + 1, 0)
        scaler.scale(loss).backward()

    if cfg.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss_accum.item(), X, Y


def train_loop(model, raw_model, optimizer, scaler, ctx, cfg: Config, rt: Runtime,
               timing: dict[str, float]) -> tuple[int, dict[str, float] | None]:
    """Executa o treino de cfg.start_iter até max_iters (ou stop_iter).

    atualiza o modelo e o otimizador, grava o CSV de métricas e o checkpoint (no processo mestre) 
    e acumula os tempos em `timing`. Retorna a última iteração alcançada e as últimas perdas de
    monitoramento.
    """
    micro_per_rank = cfg.grad_accum_per_rank
    tokens_per_iter = micro_per_rank * rt.world_size * cfg.batch_size * cfg.block_size
    iter_num = cfg.start_iter
    local_iter = 0
    running_mfu = -1.0
    last_losses = None
    interval_time, interval_iters = 0.0, 0

    def train_batch(it: int, micro: int):
        g = global_micro_batch(it, micro, rt.rank, micro_per_rank, rt.world_size)
        ids = rt.sampler.ids(g * cfg.batch_size, cfg.batch_size)
        return load_windows(rt.train_path, ids, cfg.block_size, rt.device, rt.device_type)

    if rt.master and not cfg.eval_only:
        prepare_metrics_file(
            rt.metrics_file, cfg.init_from, iter_num,
            iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters)

    X, Y = (train_batch(iter_num, 0)
            if iter_num < cfg.max_iters and not cfg.eval_only else (None, None))

    while True:
        lr = (get_lr(iter_num, cfg.warmup_iters, cfg.lr_decay_iters,
                     cfg.learning_rate, cfg.min_lr)
              if cfg.decay_lr else cfg.learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = lr

        stop_here = cfg.stop_iter is not None and iter_num >= cfg.stop_iter
        if (iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters
                or stop_here or cfg.eval_only):
            
            new_losses = evaluate_and_save_checkpoint(
                iter_num, raw_model, optimizer, scaler, ctx, cfg, rt, timing, lr,
                interval_time, interval_iters, tokens_per_iter
            )
            if new_losses is not None:
                last_losses = new_losses
                
            interval_time, interval_iters = 0.0, 0

        if cfg.eval_only or iter_num >= cfg.max_iters or stop_here:
            break

        sync(rt.device_type)
        t_start = time.perf_counter()

        lossf, X, Y = perform_training_step(
            iter_num, model, optimizer, scaler, ctx, cfg, rt, train_batch, X, Y
        )

        sync(rt.device_type)
        dt = time.perf_counter() - t_start

        timing['train_time_s'] += dt
        steady = local_iter >= cfg.timing_warmup_iters
        if steady:
            timing['steady_time_s'] += dt
            timing['steady_sq_s'] += dt * dt
            timing['steady_iters'] += 1
            interval_time += dt
            interval_iters += 1
        else:
            timing['warmup_time_s'] += dt

        if iter_num % cfg.log_interval == 0 and rt.master:
            if steady:
                mfu = raw_model.estimate_mfu(cfg.batch_size * micro_per_rank, dt)
                mfu *= 312.0 / cfg.peak_tflops
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, tempo {dt * 1000:.2f}ms, "
                  f"mfu {running_mfu * 100:.2f}%{'' if steady else ' (aquecimento)'}")

        iter_num += 1
        local_iter += 1

    return iter_num, last_losses


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Treinamento das variantes de bloco do GPT no C4")
    BOOL = argparse.BooleanOptionalAction

    p.add_argument('--block_type', default='seq', choices=['seq', 'inv', 'par'])
    p.add_argument('--init_from', default='scratch', choices=['scratch', 'resume'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--out_dir', default=None, help='padrão: out/<block_type>_s<seed>')
    p.add_argument('--dataset', default='c4', choices=list(DATASET_PRESETS),
                   help='c4 (experimento) ou shakespeare_char (depuração); define os padrões')
    p.add_argument('--data_dir', default=None)

    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--eval_only', action='store_true',
                   help='só avalia o estado atual (do zero ou do checkpoint) e encerra, sem gravar nada')
    p.add_argument('--stop_iter', type=int, default=None,
                   help='interrompe (com checkpoint) nesta iteração sem alterar o cronograma; '
                        'útil em clusters com limite de tempo')

    p.add_argument('--grad_accum', type=int, default=5 * 8)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--block_size', type=int, default=1024)

    p.add_argument('--n_layer', type=int, default=12)
    p.add_argument('--n_head', type=int, default=12)
    p.add_argument('--n_embd', type=int, default=768)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--bias', action=BOOL, default=False)

    p.add_argument('--learning_rate', type=float, default=6e-4)
    p.add_argument('--min_lr', type=float, default=6e-5)
    p.add_argument('--max_iters', type=int, default=6100)
    p.add_argument('--lr_decay_iters', type=int, default=6100)
    p.add_argument('--warmup_iters', type=int, default=200)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--decay_lr', action=BOOL, default=True)

    p.add_argument('--backend', default='nccl')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default=None, choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--compile', action=BOOL, default=True)
    p.add_argument('--allow_multi_epoch', action='store_true',
                   help='permite que o orçamento ultrapasse uma época do train.bin')
    p.add_argument('--timing_warmup_iters', type=int, default=10,
                   help='iterações iniciais de cada processo excluídas da média de tempo')
    p.add_argument('--peak_tflops', type=float, default=65.0,
               help='pico teórico da NVIDIA T4 em FP16')
    return p


def validate_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Valida a entrada do usuário. Usa parser.error (mensagem e código de
    saída 2), e não assert, que é removido quando o Python roda com -O."""
    positive = ['eval_interval', 'log_interval', 'eval_iters', 'grad_accum', 'batch_size',
                'block_size', 'n_layer', 'n_head', 'n_embd', 'max_iters', 'lr_decay_iters',
                'learning_rate', 'peak_tflops']
    for name in positive:
        if getattr(args, name) <= 0:
            p.error(f"--{name} deve ser positivo")
    if args.timing_warmup_iters < 0:
        p.error("--timing_warmup_iters não pode ser negativo")
    if args.warmup_iters < 0:
        p.error("--warmup_iters não pode ser negativo")
    if args.decay_lr and not args.warmup_iters < args.lr_decay_iters:
        p.error("--warmup_iters deve ser menor que --lr_decay_iters")
    if not 0 <= args.min_lr <= args.learning_rate:
        p.error("exige 0 <= --min_lr <= --learning_rate")
    if not 0 <= args.dropout < 1:
        p.error("--dropout deve estar em [0, 1)")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        p.error("--beta1 e --beta2 devem estar em [0, 1)")
    if args.weight_decay < 0 or args.grad_clip < 0:
        p.error("--weight_decay e --grad_clip não podem ser negativos")
    if args.n_embd % args.n_head != 0:
        p.error("--n_embd deve ser múltiplo de --n_head")
    if args.stop_iter is not None and not 0 < args.stop_iter <= args.max_iters:
        p.error("--stop_iter deve estar em (0, max_iters]")


class Config:
    """Configuração da execução: argumentos da linha de comando mais campos derivados."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.__dict__.update(vars(args))

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class Runtime:
    """Estado do processo (DDP, dispositivo, caminhos, amostrador); não é salvo."""


def main() -> None:
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    parser.set_defaults(**DATASET_PRESETS[pre.dataset])
    args = parser.parse_args()
    validate_args(parser, args)
    cfg = Config(args)
    if cfg.data_dir is None:
        cfg.data_dir = os.path.join(BASE_DIR, 'data', cfg.dataset)
    if cfg.out_dir is None:
        cfg.out_dir = os.path.join(BASE_DIR, 'out', cfg.dataset, f'{cfg.block_type}_s{cfg.seed}')

    rt = Runtime()
    rt.ddp = int(os.environ.get('RANK', -1)) != -1
    if rt.ddp:
        init_process_group(backend=cfg.backend, timeout=timedelta(minutes=60))
    try:
        run(cfg, rt)
    finally:
        if rt.ddp and dist.is_initialized():
            destroy_process_group()


def print_and_save_run_header(
    cfg: Config, rt: Runtime, tokens_per_iter: int, total_params: int,
    non_emb_params: int, planned_tokens: int, planned_epochs: float,
    n_train: int, n_val: int
) -> None:
    if rt.master:
        print("==== Iniciando treinamento ====")
        print(f"Variante: {cfg.block_type.upper()} | semente: {cfg.seed} | world_size: {rt.world_size}")
        print(f"Tokens por iteração: {tokens_per_iter:,}")
        print(f"Parâmetros: {total_params:,} totais | {non_emb_params:,} sem embeddings")
        print(f"Tokens planejados: {planned_tokens:,} | "
              f"tokens/parâmetro: {planned_tokens / total_params:.2f}")
        print(f"Janelas de treino: {n_train:,} | épocas planejadas: {planned_epochs:.4f}")

        if not cfg.eval_only:
            run_info = {
                'config': cfg.as_dict(),
                'tokens_per_iter': tokens_per_iter,
                'planned_tokens': planned_tokens,
                'planned_epochs': planned_epochs,
                'params_total': total_params,
                'params_non_embedding': non_emb_params,
                'tokens_per_param': planned_tokens / total_params,
                'train_windows': n_train,
                'val_windows': n_val,
                'world_size': rt.world_size,
                'environment': environment_info(),
            }
            name = ('config.json' if cfg.init_from == 'scratch'
                    else f'config_resume_{cfg.start_iter}.json')
            write_json(os.path.join(cfg.out_dir, name), run_info)


def save_final_summary(
    cfg: Config, rt: Runtime, timing: dict[str, float], total_params: int,
    non_emb_params: int, final_iter: int, last_losses: dict[str, float] | None,
    tokens_per_iter: int, n_train: int
) -> None:
    if rt.master and not cfg.eval_only and final_iter >= cfg.max_iters:
        n = timing['steady_iters']
        mean_s = timing['steady_time_s'] / n if n else float('nan')
        var = timing['steady_sq_s'] / n - mean_s ** 2 if n else float('nan')
        std_s = math.sqrt(max(var, 0.0)) if n else float('nan')
        iters_per_epoch = n_train / (cfg.grad_accum * cfg.batch_size)
        summary = {
            'dataset': cfg.dataset,
            'block_type': cfg.block_type,
            'seed': cfg.seed,
            'iters': final_iter,
            'tokens_trained': final_iter * tokens_per_iter,
            'epochs_trained': final_iter / iters_per_epoch,
            'params_total': total_params,
            'params_non_embedding': non_emb_params,
            'train_time_s': timing['train_time_s'],
            'warmup_time_s': timing['warmup_time_s'],
            'eval_time_s': timing['eval_time_s'],
            'steady_iters': n,
            'step_time_ms_mean': mean_s * 1000,
            'step_time_ms_std': std_s * 1000,
            'tokens_per_s': tokens_per_iter / mean_s if n else float('nan'),
            'iters_per_epoch': iters_per_epoch,
            'time_per_epoch_s': mean_s * iters_per_epoch,
            'monitor_val_loss_final': last_losses['val'] if last_losses else None,
            'world_size': rt.world_size,
            'peak_tflops': cfg.peak_tflops,
            'gpu': environment_info().get('gpu_name'),
        }
        write_json(os.path.join(cfg.out_dir, 'summary.json'), summary)
        print(f"Resumo salvo em {os.path.join(cfg.out_dir, 'summary.json')}")


def run(cfg: Config, rt: Runtime) -> None:
    if rt.ddp:
        rt.rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        rt.world_size = int(os.environ['WORLD_SIZE'])
        rt.device = f'cuda:{local_rank}'
        torch.cuda.set_device(rt.device)
    else:
        rt.rank, local_rank, rt.world_size = 0, 0, 1
        rt.device = cfg.device
    rt.master = rt.rank == 0
    rt.device_type = 'cuda' if 'cuda' in str(rt.device) else 'cpu'
    if cfg.dtype is None:
        cfg.dtype = default_dtype(rt.device_type)

    if cfg.grad_accum % rt.world_size != 0:
        raise ValueError(f"grad_accum ({cfg.grad_accum}) deve ser múltiplo do "
                         f"número de processos ({rt.world_size})")
    cfg.grad_accum_per_rank = cfg.grad_accum // rt.world_size
    tokens_per_iter = cfg.grad_accum * cfg.batch_size * cfg.block_size

    rt.checkpoint_path = os.path.join(cfg.out_dir, 'ckpt.pt')
    rt.metrics_file = os.path.join(cfg.out_dir, f'metrics_{cfg.block_type}.csv')
    rt.train_path = os.path.join(cfg.data_dir, 'train.bin')
    val_path = find_val_file(cfg.data_dir)
    for path in (rt.train_path, val_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} não encontrado. Prepare o dataset com: "
                f"{DATASET_HINTS[cfg.dataset]}"
            )
    if rt.master:
        os.makedirs(cfg.out_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = autocast_ctx(rt.device_type, cfg.dtype)

    cfg.data_fingerprint = data_fingerprint([rt.train_path, val_path])
    n_train = count_windows(rt.train_path, cfg.block_size)
    n_val = count_windows(val_path, cfg.block_size)
    n_monitor = cfg.eval_iters * cfg.batch_size
    if n_monitor > n_train:
        raise ValueError(f"Monitoramento ({n_monitor:,} janelas) maior que o treino "
                         f"({n_train:,} janelas). Reduza --eval_iters ou --batch_size.")
    if n_monitor >= n_val:
        raise ValueError(f"Monitoramento ({n_monitor:,} janelas) não deixa janelas de "
                         f"validação ({n_val:,}) para a avaliação final.")
    cfg.val_monitor_windows = n_monitor
    rt.sampler = TrainSampler(n_train, cfg.seed)
    rt.monitor = {
        'train': (rt.train_path,
                  np.random.default_rng(MONITOR_SEED).choice(n_train, n_monitor, replace=False)),
        'val': (val_path, np.arange(n_monitor)),
    }

    model_args = dict(
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        block_size=cfg.block_size, bias=cfg.bias,
        vocab_size=dataset_vocab_size(cfg.data_dir),
        dropout=cfg.dropout, block_type=cfg.block_type,
    )
    checkpoint = None
    cfg.start_iter = 0
    timing = new_timing()

    if cfg.init_from == 'resume':
        if not os.path.exists(rt.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint não encontrado: {rt.checkpoint_path}")
        checkpoint = torch.load(rt.checkpoint_path, map_location=rt.device, weights_only=False)
        saved = checkpoint.get('config', {})
        diffs = [k for k in RESUME_LOCKED_KEYS
                 if saved.get(k, '<ausente>') != getattr(cfg, k)]
        if diffs:
            raise ValueError(
                "Configuração diferente da do checkpoint em: "
                + ", ".join(f"{k} ({saved.get(k, '<ausente>')!r} -> {getattr(cfg, k)!r})"
                            for k in diffs))
        model_args = checkpoint['model_args']
        cfg.start_iter = checkpoint['iter_num']
        timing.update(checkpoint.get('timing', {}))
        if rt.master:
            print(f"Retomando a partir da iteração {cfg.start_iter}.")
    cfg.model_args = model_args

    model = GPT(GPTConfig(**model_args))
    if checkpoint is not None:
        model.load_state_dict(strip_compile_prefix(checkpoint['model']))
    model.to(rt.device)

    total_params, non_emb_params = count_params(model)
    planned_tokens = cfg.max_iters * tokens_per_iter
    planned_epochs = cfg.max_iters * cfg.grad_accum * cfg.batch_size / n_train

    if planned_epochs > 1 and not cfg.allow_multi_epoch:
        raise ValueError(
            f"O orçamento ({cfg.max_iters * cfg.grad_accum * cfg.batch_size:,} janelas) excede "
            f"as {n_train:,} janelas do train.bin. Reduza max_iters ou use --allow_multi_epoch.")

    print_and_save_run_header(
        cfg, rt, tokens_per_iter, total_params, non_emb_params,
        planned_tokens, planned_epochs, n_train, n_val
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.dtype == 'float16' and rt.device_type == 'cuda'))
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), rt.device_type)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

    if cfg.compile:
        if rt.master:
            print("Compilando o modelo...")
        model = torch.compile(model)
    if rt.ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if rt.ddp else model
    if cfg.compile:
        raw_model = raw_model._orig_mod

    if checkpoint is not None:
        states = checkpoint.get('rng_states')
        if states is not None and len(states) == rt.world_size:
            set_rng_state(states[rt.rank])
        elif rt.master:
            print("Aviso: estados aleatórios indisponíveis para este world_size; "
                  "a ordem dos dados continua exata.")
        checkpoint = None

    final_iter, last_losses = train_loop(
        model, raw_model, optimizer, scaler, ctx, cfg, rt, timing)

    save_final_summary(
        cfg, rt, timing, total_params, non_emb_params, final_iter,
        last_losses, tokens_per_iter, n_train
    )


if __name__ == "__main__":
    main()
