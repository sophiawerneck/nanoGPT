#!/usr/bin/env python3
"""
Gera amostras de texto a partir de um modelo treinado (ou GPT-2 original).
"""
from __future__ import annotations

import os
import pickle
import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import tiktoken
from simple_parsing import ArgumentParser, choice

from model import GPTConfig, GPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SampleConfig:
    """Configurações para amostragem de texto do modelo."""
    init_from: str = choice("resume", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", default="resume")
    out_dir: str = 'out'
    start: str = '\n'  # Pode ser texto direto ou "FILE:prompt.txt"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    seed: int = 1337
    device: str = 'cuda'
    dtype: str | None = None
    compile: bool = False

    def __post_init__(self):
        if self.num_samples <= 0:
            raise ValueError("--num_samples deve ser positivo")
        if self.max_new_tokens <= 0:
            raise ValueError("--max_new_tokens deve ser positivo")


def main():
    parser = ArgumentParser(description="Geração de texto (Sampling) usando o modelo GPT.")
    parser.add_arguments(SampleConfig, dest="cfg")
    args = parser.parse_args()
    cfg: SampleConfig = args.cfg

    device = torch.device(cfg.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError("--device deve indicar 'cpu' ou 'cuda' (por exemplo, cuda:0).")
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("Foi solicitado CUDA, mas CUDA não está disponível no sistema.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Dispositivo {device} inexistente; GPUs disponíveis: {torch.cuda.device_count()}."
            )
        torch.cuda.set_device(device)

    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = None
    if cfg.init_from == 'resume':
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
        
        logger.info("Carregando checkpoint de %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
                
        model.load_state_dict(state_dict)
    
    elif cfg.init_from.startswith('gpt2'):
        logger.info("Inicializando a partir do modelo HuggingFace pré-treinado: %s", cfg.init_from)
        model = GPT.from_pretrained(cfg.init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"Origem desconhecida: {cfg.init_from}")

    model.eval()
    model.to(device)

    saved_dtype = checkpoint.get('config', {}).get('dtype') if checkpoint else None
    if cfg.dtype is not None:
        dtype = cfg.dtype
    elif device.type == 'cpu':
        dtype = 'float32'
    elif saved_dtype in ('float32', 'bfloat16', 'float16'):
        dtype = saved_dtype
    else:
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'

    if device.type == 'cpu' and dtype != 'float32':
        raise ValueError("A geração em CPU utiliza float32. Altere o --dtype ou use CUDA.")
    if device.type == 'cuda' and dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError("A GPU selecionada não oferece suporte a bfloat16.")

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    autocast_enabled = device.type == 'cuda' and dtype in ('float16', 'bfloat16')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if autocast_enabled else nullcontext()

    if cfg.compile:
        logger.info("Compilando o modelo (isso pode demorar a primeira vez)...")
        model = torch.compile(model)

    load_meta = False
    meta_path = ""
    if cfg.init_from == 'resume' and checkpoint and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        logger.info("Carregando o vocabulário (meta) de %s...", meta_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos.get(i, '') for i in l])
    else:
        logger.info("Nenhum meta.pkl encontrado, assumindo tokenizador GPT-2 padrão...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    start_text = cfg.start
    if start_text.startswith('FILE:'):
        file_path = start_text[5:]
        with open(file_path, 'r', encoding='utf-8') as f:
            start_text = f.read()
            
    start_ids = encode(start_text)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    logger.info(
        "Gerando %d amostras (max %d tokens, temp %.2f, top_k %d)...",
        cfg.num_samples, cfg.max_new_tokens, cfg.temperature, cfg.top_k
    )
    
    with torch.no_grad():
        with ctx:
            for k in range(cfg.num_samples):
                y = model.generate(x, cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
                
                print(decode(y[0].tolist()))
                print('---------------')


if __name__ == '__main__':
    main()
