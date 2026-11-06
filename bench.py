
"""
A much shorter version of train.py for benchmarking inference latency.
"""
import os
import time
import json
import argparse
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark inference latency")
parser.add_argument('--out_dir', type=str, default='out')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default=None,
                    choices=['float32', 'bfloat16', 'float16'])
parser.add_argument('--compile', action='store_true')
parser.add_argument('--max_new_tokens', type=int, default=200)
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--seed', type=int, default=1337)
args = parser.parse_args()

if args.max_new_tokens <= 0:
    parser.error("--max_new_tokens deve ser positivo")

if args.num_samples <= 0:
    parser.error("--num_samples deve ser positivo")

out_dir = args.out_dir
device = args.device
compile = args.compile
max_new_tokens = args.max_new_tokens
num_samples = args.num_samples
seed = args.seed
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# model init from checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# Utiliza a precisão do treinamento como padrão, permitindo sobrescrita.
saved_dtype = checkpoint.get('config', {}).get('dtype')

if args.dtype is not None:
    dtype = args.dtype
elif saved_dtype in ('float32', 'bfloat16', 'float16'):
    dtype = saved_dtype
elif device_type == 'cuda':
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
else:
    dtype = 'float32'

ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype
)

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'

for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# dummy prompt for latency testing (10 tokens)
prompt_length = 10
x = torch.randint(
    0, gptconf.vocab_size,
    (1, prompt_length),
    dtype=torch.long,
    device=device
)

# -----------------------------------------------------------------------------
# benchmarking
generate_times = []

print("Warming up...")
with torch.no_grad():
    with ctx:
        _ = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=200
        )

if device_type == 'cuda':
    torch.cuda.synchronize()

print("Benchmarking...")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            t0 = time.perf_counter()

            y = model.generate(
                x,
                max_new_tokens,
                temperature=0.8,
                top_k=200
            )

            if device_type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            dt = t1 - t0
            generate_times.append(dt)
            print(f"{k+1}/{num_samples} time: {dt:.3f}s")

# stats
mean_time = sum(generate_times) / len(generate_times)

std_time = (
    sum((t - mean_time) ** 2 for t in generate_times) / len(generate_times)
) ** 0.5

time_per_token = mean_time / max_new_tokens
tokens_per_sec = max_new_tokens / mean_time

print(f"Mean latency time: {mean_time:.3f} s ({mean_time * 1000:.2f} ms)")
print(f"Standard deviation: {std_time:.3f} s ({std_time * 1000:.2f} ms)")
print(f"Time per token: {time_per_token * 1000:.2f} ms")
print(f"Throughput: {tokens_per_sec:.2f} tokens/s")

# dump for summarize.py
bench_metrics = {
    "inference_latency_mean_s": mean_time,
    "inference_latency_std_s": std_time,
    "inference_time_per_token_ms": time_per_token * 1000,
    "inference_tokens_per_sec": tokens_per_sec,
    "inference_max_new_tokens": max_new_tokens,

    # Configurações utilizadas no benchmark
    "inference_device": str(device),
    "inference_dtype": dtype,
    "inference_autocast_enabled": device_type == 'cuda',
    "inference_compile": compile,
    "inference_num_samples": num_samples,
    "inference_prompt_length": prompt_length,
    "inference_seed": seed
}

bench_out_path = os.path.join(out_dir, 'benchmark.json')

with open(bench_out_path, 'w', encoding='utf-8') as f:
    json.dump(bench_metrics, f, indent=4)
