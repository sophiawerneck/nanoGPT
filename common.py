"""Funções compartilhadas entre preparação, treino, avaliação e benchmark."""

import os
import json
import platform
import subprocess
from contextlib import nullcontext

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_DTYPE = np.uint16  


def count_tokens(path):
    return os.path.getsize(path) // np.dtype(TOKEN_DTYPE).itemsize


def count_windows(path, block_size):
    """Número de janelas NÃO sobrepostas de block_size tokens (+1 para o alvo)."""
    return (count_tokens(path) - 1) // block_size


def load_windows(path, window_ids, block_size, device, device_type):
    """Carrega as janelas indicadas. A janela w cobre os tokens
    [w*block_size, w*block_size + block_size] (entrada e alvo deslocado)."""
    # Recria o memmap a cada chamada para evitar o vazamento de memória
    # documentado no nanoGPT original.
    data = np.memmap(path, dtype=TOKEN_DTYPE, mode='r')
    xs, ys = [], []
    for w in window_ids:
        start = int(w) * block_size
        chunk = torch.from_numpy(data[start:start + block_size + 1].astype(np.int64))
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.stack(xs)
    y = torch.stack(ys)
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def default_dtype(device_type):
    if device_type == 'cuda':
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    return 'float32'


def autocast_ctx(device_type, dtype):
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[dtype]
    if device_type == 'cpu':
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def sync(device_type):
    if device_type == 'cuda':
        torch.cuda.synchronize()


def strip_compile_prefix(state_dict):
    prefix = '_orig_mod.'
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def load_model_from_checkpoint(ckpt_path, device):
    from model import GPTConfig, GPT

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict(strip_compile_prefix(ckpt['model']))
    model.to(device)
    model.eval()
    return model, ckpt


def count_params(model):
    """Retorna (total, não-embedding).

    total: parâmetros únicos (wte e lm_head são compartilhados e contados uma
    vez), convenção usada por Hoffmann et al. (2022, Chinchilla).
    não-embedding: exclui embeddings de token e de posição, convenção de
    Kaplan et al. (2020).
    """
    total = sum(p.numel() for p in model.parameters())
    embedding = (
        model.transformer.wte.weight.numel()
        + model.transformer.wpe.weight.numel()
    )
    return total, total - embedding


def _git(args):
    try:
        out = subprocess.run(
            ['git'] + args, cwd=BASE_DIR, capture_output=True,
            text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info():
    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'git_commit': _git(['rev-parse', 'HEAD']),
        'git_dirty': bool(_git(['status', '--porcelain'])),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name()
        info['gpu_count'] = torch.cuda.device_count()
    return info


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
