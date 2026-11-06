"""
Sample from a trained model
"""
import os
import argparse
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Sample from a trained model")
parser.add_argument('--init_from', type=str, default='resume')
parser.add_argument('--out_dir', type=str, default='out')
parser.add_argument('--start', type=str, default='\n')
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_k', type=int, default=200)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default=None)
parser.add_argument('--compile', action='store_true')
args = parser.parse_args()

init_from = args.init_from
out_dir = args.out_dir
start = args.start
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
device = args.device
dtype = args.dtype if args.dtype is not None else ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
compile = args.compile
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
generate_times = []

torch.cuda.synchronize()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            start_time = time.time()
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            end_time = time.time()

            elapsed_time = end_time - start_time
            generate_times.append(elapsed_time)

            print(decode(y[0].tolist()))
            print(f"Latency time: {elapsed_time:.3f} s ({elapsed_time * 1000:.2f} ms)")
            print('---------------')

mean_time = sum(generate_times) / len(generate_times)
std_time = (sum((t - mean_time) ** 2 for t in generate_times) / len(generate_times)) ** 0.5
time_per_token = mean_time / max_new_tokens

print(f"Mean latency time: {mean_time:.2f} s ({mean_time * 1000:.2f} ms)")
print(f"Standard deviation of latency time: {std_time:.2f} s ({std_time * 1000:.2f} ms)")
print(f"Time per token: {time_per_token:.2f} s ({time_per_token * 1000:.2f} ms)")