"""Treinamento das variantes de bloco (seq, inv, par) do GPT (~124M) no C4.

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
- Amostragem SEM reposição: o train.bin é dividido em janelas não sobrepostas
  de block_size tokens, percorridas numa permutação determinada pela semente.
  Cada token é visto no máximo uma vez por época (passagem única, como no
  GPT-3 e no Chinchilla). Mesma semente = mesma ordem de dados em todas as
  variantes (comparação pareada).
- Monitoramento em subconjuntos FIXOS de treino e validação; a avaliação final
  (evaluate.py) usa janelas de validação disjuntas das de monitoramento.
- Tempo de passo medido com sincronização da GPU, excluindo a avaliação e as
  primeiras iterações de cada processo (compilação e aquecimento).
"""

import os
import time
import math
import random
import csv
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from common import (
    BASE_DIR, count_windows, load_windows, default_dtype, autocast_ctx, sync,
    strip_compile_prefix, count_params, environment_info, write_json,
)

# Semente fixa dos subconjuntos de monitoramento: igual para todas as
# variantes e sementes, para que as curvas sejam comparáveis.
MONITOR_SEED = 12345

# Hiperparâmetros que não podem mudar numa retomada.
RESUME_LOCKED_KEYS = [
    'batch_size', 'block_size', 'grad_accum', 'learning_rate', 'min_lr',
    'max_iters', 'lr_decay_iters', 'warmup_iters', 'weight_decay', 'beta1',
    'beta2', 'grad_clip', 'decay_lr', 'seed', 'eval_iters', 'eval_interval',
]

METRIC_COLUMNS = [
    'iter',
    'train_loss',
    'val_loss',
    'train_ppl',
    'val_ppl',
    'lr',
    'step_time_ms', # média dos passos estáveis desde a última avaliação
    'tokens_per_s', # throughput de treino no mesmo intervalo (por processo x world_size)
    'train_time_s', # tempo acumulado de treino (sem avaliação)
    'eval_time_s', # tempo acumulado gasto em avaliação
]

class TrainSampler:
    """Ordem sem reposição sobre as janelas de treino.

    A amostra global de índice s pertence à época s // n e ocupa a posição
    s % n de uma permutação gerada com (seed, época). O lote de qualquer
    iteração pode ser recalculado a partir de iter_num.
    """

    def __init__(self, num_windows, seed):
        self.n = num_windows
        self.seed = seed
        self._epoch = -1
        self._perm = None

    def _perm_for(self, epoch):
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, epoch])
            self._perm = rng.permutation(self.n)
            self._epoch = epoch
        return self._perm

    def ids(self, start, count):
        out = np.empty(count, dtype=np.int64)
        for j in range(count):
            epoch, pos = divmod(start + j, self.n)
            out[j] = self._perm_for(epoch)[pos]
        return out


@torch.no_grad()
def estimate_loss(model, ctx, monitor, batch_size, block_size, device, device_type):
    """Loss média em subconjuntos fixos (sem sorteio) de treino e validação."""
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

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def collect_rng_states(ddp):
    """Estados aleatórios de todos os ranks (relevantes se dropout > 0)."""
    local = get_rng_state()
    if ddp:
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
        return states
    return [local]


def save_checkpoint(path, checkpoint):
    tmp = path + '.tmp'
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def new_timing():
    return {
        'train_time_s': 0.0, # todos os passos (inclui compilação), sem avaliação
        'warmup_time_s': 0.0, # passos excluídos da média estável
        'steady_time_s': 0.0,
        'steady_sq_s': 0.0,
        'steady_iters': 0,
        'eval_time_s': 0.0,
    }


def prepare_metrics_file(metrics_file, init_from, resume_iter, reevaluates_resume_iter):
    """Na retomada, mantém as linhas anteriores. A linha da própria iteração de
    retomada só é descartada se ela for avaliada de novo."""
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

def train_loop(model, raw_model, optimizer, scaler, ctx, cfg, rt, timing):
    micro_per_rank = cfg.grad_accum_per_rank
    micro_total = micro_per_rank * rt.world_size
    tokens_per_iter = micro_total * cfg.batch_size * cfg.block_size
    iter_num = cfg.start_iter
    local_iter = 0
    running_mfu = -1.0
    last_losses = None
    interval_time, interval_iters = 0.0, 0

    def train_batch(it, micro):
        g = it * micro_total + micro * rt.world_size + rt.rank
        ids = rt.sampler.ids(g * cfg.batch_size, cfg.batch_size)
        return load_windows(rt.train_path, ids, cfg.block_size, rt.device, rt.device_type)

    if rt.master:
        prepare_metrics_file(
            rt.metrics_file, cfg.init_from, iter_num,
            iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters)

    X, Y = train_batch(iter_num, 0) if iter_num < cfg.max_iters else (None, None)

    while True:
        lr = (get_lr(iter_num, cfg.warmup_iters, cfg.lr_decay_iters,
                     cfg.learning_rate, cfg.min_lr)
              if cfg.decay_lr else cfg.learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = lr

        stop_here = cfg.stop_iter is not None and iter_num >= cfg.stop_iter
        if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters or stop_here:
            if rt.ddp:
                dist.barrier()
            rng_states = collect_rng_states(rt.ddp)

            if rt.master:
                sync(rt.device_type)
                t_eval = time.perf_counter()
                losses = estimate_loss(raw_model, ctx, rt.monitor, cfg.batch_size,
                                       cfg.block_size, rt.device, rt.device_type)
                sync(rt.device_type)
                timing['eval_time_s'] += time.perf_counter() - t_eval
                last_losses = losses

                if interval_iters > 0:
                    step_s = interval_time / interval_iters
                    step_ms, tok_s = step_s * 1000, tokens_per_iter / step_s
                else:
                    step_ms, tok_s = float('nan'), float('nan')

                print(f"step {iter_num}: train loss {losses['train']:.4f} | "
                      f"val loss {losses['val']:.4f} | val ppl {math.exp(losses['val']):.2f} | "
                      f"passo {step_ms:.1f} ms | {tok_s:,.0f} tok/s")

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

                # não há seleção de modelo; o modelo avaliado é o final (iteração max_iters).
                if iter_num > 0:
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

            interval_time, interval_iters = 0.0, 0
            if rt.ddp:
                dist.barrier()

        if iter_num == 0 and cfg.eval_only:
            break
        if iter_num >= cfg.max_iters or stop_here:
            break

        sync(rt.device_type)
        t_start = time.perf_counter()

        loss_accum = torch.zeros((), device=rt.device)
        for micro in range(micro_per_rank):
            if rt.ddp:
                model.require_backward_grad_sync = (micro == micro_per_rank - 1)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / micro_per_rank
            loss_accum += loss.detach().float()
            if micro + 1 < micro_per_rank:
                X, Y = train_batch(iter_num, micro + 1)
            elif iter_num + 1 < cfg.max_iters:
                X, Y = train_batch(iter_num + 1, 0)
            scaler.scale(loss).backward()

        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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
            lossf = loss_accum.item()
            if steady:
                mfu = raw_model.estimate_mfu(cfg.batch_size * micro_per_rank, dt)
                mfu *= 312.0 / cfg.peak_tflops
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, tempo {dt * 1000:.2f}ms, "
                  f"mfu {running_mfu * 100:.2f}%{'' if steady else ' (aquecimento)'}")

        iter_num += 1
        local_iter += 1

    return iter_num, last_losses
  
def parse_args():
    p = argparse.ArgumentParser(description="Treinamento das variantes de bloco do GPT no C4")
    BOOL = argparse.BooleanOptionalAction

    p.add_argument('--block_type', default='seq', choices=['seq', 'inv', 'par'])
    p.add_argument('--init_from', default='scratch', choices=['scratch', 'resume'])
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--out_dir', default=None, help='padrão: out/<block_type>_s<seed>')
    p.add_argument('--data_dir', default=os.path.join(BASE_DIR, 'data', 'c4'))

    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--log_interval', type=int, default=1)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--eval_only', action='store_true')
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
    p.add_argument('--peak_tflops', type=float, default=312.0,
                   help='pico teórico da GPU no dtype usado (T4 bf16 = 65)')
    return p.parse_args()


class Config:
    def __init__(self, args):
        self.__dict__.update(vars(args))

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class Runtime:
    pass


def main():
    cfg = Config(parse_args())
    if cfg.out_dir is None:
        cfg.out_dir = os.path.join(BASE_DIR, 'out', f'{cfg.block_type}_s{cfg.seed}')

    rt = Runtime()
    rt.ddp = int(os.environ.get('RANK', -1)) != -1
    if rt.ddp:
        # Timeout maior: só o rank 0 avalia enquanto os demais aguardam na barreira.
        init_process_group(backend=cfg.backend, timeout=timedelta(minutes=60))
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

    assert cfg.grad_accum % rt.world_size == 0, "grad_accum deve ser múltiplo de world_size"
    cfg.grad_accum_per_rank = cfg.grad_accum // rt.world_size
    tokens_per_iter = cfg.grad_accum * cfg.batch_size * cfg.block_size

    rt.checkpoint_path = os.path.join(cfg.out_dir, 'ckpt.pt')
    rt.metrics_file = os.path.join(cfg.out_dir, f'metrics_{cfg.block_type}.csv')
    rt.train_path = os.path.join(cfg.data_dir, 'train.bin')
    val_path = os.path.join(cfg.data_dir, 'validation.bin')
    for path in (rt.train_path, val_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} não encontrado. Rode prepare_c4.py antes.")

    if rt.master:
        os.makedirs(cfg.out_dir, exist_ok=True)

    # Semente: inicialização do modelo (idêntica em todos os ranks; o DDP
    # também a difunde a partir do rank 0) e ordem dos dados.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = autocast_ctx(rt.device_type, cfg.dtype)

    n_train = count_windows(rt.train_path, cfg.block_size)
    n_val = count_windows(val_path, cfg.block_size)
    n_monitor = cfg.eval_iters * cfg.batch_size
    if n_monitor >= n_val:
        raise ValueError("Validação pequena demais para o monitoramento e a avaliação final.")
    cfg.val_monitor_windows = n_monitor
    rt.sampler = TrainSampler(n_train, cfg.seed)
    rt.monitor = {
        'train': (rt.train_path,
                  np.random.default_rng(MONITOR_SEED).choice(n_train, n_monitor, replace=False)),
        'val': (val_path, np.arange(n_monitor)),
    }

    model_args = dict(
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        block_size=cfg.block_size, bias=cfg.bias, vocab_size=50304,
        dropout=cfg.dropout, block_type=cfg.block_type,
    )
    checkpoint = None
    cfg.start_iter = 0
    timing = new_timing()

    if cfg.init_from == 'resume':
        if not os.path.exists(rt.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint não encontrado: {rt.checkpoint_path}")
        checkpoint = torch.load(rt.checkpoint_path, map_location=rt.device, weights_only=False)
        if checkpoint['model_args'].get('block_type') != cfg.block_type:
            raise ValueError("O block_type do checkpoint não corresponde à variante solicitada.")
        saved = checkpoint.get('config', {})
        diffs = [k for k in RESUME_LOCKED_KEYS
                 if k in saved and saved[k] != getattr(cfg, k)]
        if diffs:
            raise ValueError(f"Hiperparâmetros diferentes do checkpoint: {diffs}")
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

    if rt.master:
        print("==== Iniciando treinamento ====")
        print(f"Variante: {cfg.block_type.upper()} | semente: {cfg.seed} | world_size: {rt.world_size}")
        print(f"Tokens por iteração: {tokens_per_iter:,}")
        print(f"Parâmetros: {total_params:,} totais | {non_emb_params:,} sem embeddings")
        print(f"Tokens planejados: {planned_tokens:,} | "
              f"tokens/parâmetro: {planned_tokens / total_params:.2f}")
        print(f"Janelas de treino: {n_train:,} | épocas planejadas: {planned_epochs:.4f}")

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
        name = 'config.json' if cfg.init_from == 'scratch' else f'config_resume_{cfg.start_iter}.json'
        write_json(os.path.join(cfg.out_dir, name), run_info)

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
            # A ordem dos dados não depende disso; só afetaria dropout > 0.
            print("Aviso: estados aleatórios indisponíveis para este world_size; "
                  "a ordem dos dados continua exata.")
        checkpoint = None

    final_iter, last_losses = train_loop(
        model, raw_model, optimizer, scaler, ctx, cfg, rt, timing)

    if rt.master and final_iter >= cfg.max_iters:
        n = timing['steady_iters']
        mean_s = timing['steady_time_s'] / n if n else float('nan')
        var = timing['steady_sq_s'] / n - mean_s ** 2 if n else float('nan')
        std_s = math.sqrt(max(var, 0.0)) if n else float('nan')
        iters_per_epoch = n_train / (cfg.grad_accum * cfg.batch_size)
        summary = {
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

    if rt.ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
