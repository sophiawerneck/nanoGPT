#!/usr/bin/env python3
import os
import json
import time
import argparse
from contextlib import nullcontext
import logging

import torch
from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--dtype', type=str, default=None,
        choices=['float32', 'bfloat16', 'float16']
    )
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens deve ser positivo")
    if args.num_samples <= 0:
        parser.error("--num_samples deve ser positivo")

    return args


def main():
    args = parse_args()
    out_dir = args.out_dir

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(out_dir, 'bench.log'),
                mode='a',
                encoding='utf-8'
            )
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError("--device deve indicar 'cpu' ou 'cuda' (por exemplo, cuda:0).")
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("Foi solicitado CUDA, mas CUDA não está disponível.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Dispositivo {device} inexistente; GPUs disponíveis: "
                f"{torch.cuda.device_count()}."
            )
        if device.index is not None:
            torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    saved_dtype = checkpoint.get('config', {}).get('dtype')
    if args.dtype is not None:
        dtype = args.dtype
    elif device.type == 'cpu':
        dtype = 'float32'
    elif saved_dtype in ('float32', 'bfloat16', 'float16'):
        dtype = saved_dtype
    else:
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'

    if device.type == 'cpu' and dtype != 'float32':
        raise ValueError(
            "Neste benchmark, a execução em CPU utiliza float32. "
            "Use --dtype float32 ou execute em CUDA."
        )
    if device.type == 'cuda' and dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError("A GPU selecionada não oferece suporte a bfloat16.")

    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[dtype]
    autocast_enabled = device.type == 'cuda' and dtype in ('float16', 'bfloat16')
    ctx = (
        torch.amp.autocast(device_type='cuda', dtype=ptdtype)
        if autocast_enabled else nullcontext()
    )

    if 'model_args' not in checkpoint or 'model' not in checkpoint:
        raise KeyError("O checkpoint precisa conter 'model_args' e 'model'.")

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if args.compile:
        logger.info("Compiling model...") 
        model = torch.compile(model)

    prompt_length = 10
    prompt_generator = torch.Generator(device=device)
    prompt_generator.manual_seed(args.seed)
    x = torch.randint(
        0,
        gptconf.vocab_size,
        (1, prompt_length),
        dtype=torch.long,
        device=device,
        generator=prompt_generator
    )

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    logger.info("Warming up...")
    with torch.no_grad():
        with ctx:
            _ = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_k=200
            )
    synchronize()

    logger.info("Benchmarking...")
    generate_times = []
    generated_token_counts = []

    with torch.no_grad():
        with ctx:
            for sample_idx in range(args.num_samples):
                sample_seed = args.seed + sample_idx
                torch.manual_seed(sample_seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(sample_seed)
                synchronize()
                start_time = time.perf_counter()

                y = model.generate(
                    x,
                    args.max_new_tokens,
                    temperature=0.8,
                    top_k=200
                )

                synchronize()
                elapsed = time.perf_counter() - start_time
                generated_tokens = int(y.shape[1] - x.shape[1])
                if generated_tokens <= 0:
                    raise RuntimeError("model.generate() não retornou nenhum token além do prompt.")

                generate_times.append(elapsed)
                generated_token_counts.append(generated_tokens)
                logger.info(f"{sample_idx + 1}/{args.num_samples} time: {elapsed:.3f}s")
    mean_time = sum(generate_times) / len(generate_times)
    std_time = (
        sum((elapsed - mean_time) ** 2 for elapsed in generate_times)
        / len(generate_times)
    ) ** 0.5

    total_generated_tokens = sum(generated_token_counts)
    total_generation_time = sum(generate_times)
    tokens_per_sec = total_generated_tokens / total_generation_time

    logger.info(f"Mean latency time: {mean_time:.3f} s ({mean_time * 1000:.2f} ms)")
    logger.info(f"Standard deviation: {std_time:.3f} s ({std_time * 1000:.2f} ms)")
    logger.info(f"Throughput: {tokens_per_sec:.2f} tokens/s")
    bench_metrics = {
        'inference_latency_mean_s': mean_time,
        'inference_latency_std_s': std_time,
        'inference_tokens_per_sec': tokens_per_sec,
        'inference_max_new_tokens': args.max_new_tokens,
        'inference_device': str(device),
        'inference_gpu': (
            torch.cuda.get_device_name(device) if device.type == 'cuda' else None
        ),
        'inference_dtype': dtype,
        'inference_autocast_enabled': autocast_enabled,
        'inference_tf32_enabled': (
            bool(torch.backends.cuda.matmul.allow_tf32) if device.type == 'cuda' else False
        ),
        'inference_compile': args.compile,
        'inference_num_samples': args.num_samples,
        'inference_prompt_length': prompt_length,
        'inference_seed': args.seed,
    }

    bench_out_path = os.path.join(out_dir, 'benchmark.json')
    with open(bench_out_path, 'w', encoding='utf-8') as file:
        json.dump(bench_metrics, file, indent=4, ensure_ascii=False)

    logger.info(f"Benchmark salvo em: {bench_out_path}")


if __name__ == '__main__':
    main()
