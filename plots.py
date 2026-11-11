import re
import math
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODEL_TITLES = {
    "seq": "Treinamento Baseline",
    "inv": "Treinamento Invertido",
    "par": "Treinamento Paralelo",
    "par_shared": "Treinamento Paralelo Compartilhado",
}


def get_metrics_from_log(log_file):
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "perplexity": [],
        "gap": [],
        "iteration": [],
        "time_iteration": [],
        "time_per_iteration_ms": []
    }

    loss_pattern = re.compile(
        r'step\s+(\d+):\s*train loss\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))'
        r'\s*(?:\||,)\s*val (?:monitor )?loss\s+'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))'
    )
    time_pattern = re.compile(
        r'iter\s+(\d+):\s*loss\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+),'
        r'\s*tempo\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))ms'
    )

    with open(log_file, encoding="utf-8") as log:
        for line in log:
            match = loss_pattern.search(line)
            if match:
                iteration, train_loss, val_loss = match.groups()
                train_loss, val_loss = float(train_loss), float(val_loss)
                metrics["iteration"].append(int(iteration))
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["perplexity"].append(math.exp(val_loss))
                metrics["gap"].append(val_loss - train_loss)
                continue

            match = time_pattern.search(line)
            if match:
                iteration, elapsed_ms = match.groups()
                metrics["time_iteration"].append(int(iteration))
                metrics["time_per_iteration_ms"].append(float(elapsed_ms))

    return metrics


def get_metrics_from_seed_logs(out_dir, seeds, type_model):
    if not seeds:
        raise ValueError("É necessário fornecer ao menos uma seed.")
    if type_model not in MODEL_TITLES:
        raise ValueError(f"Tipo de bloco desconhecido: {type_model}.")

    runs = []
    for seed in seeds:
        log_file = os.path.join(out_dir, f"{type_model}_s{seed}", "log.txt")
        runs.append(get_metrics_from_log(log_file))
    return runs


def _aggregate_metric(runs, x_key, y_key, smooth=False):
    values_by_run = [
        dict(zip(run[x_key], run[y_key]))
        for run in runs
    ]
    if not values_by_run or any(not values for values in values_by_run):
        raise ValueError(f"Nenhum dado disponível para agregar {y_key}.")

    common_steps = sorted(set.intersection(*(set(values) for values in values_by_run)))
    if not common_steps:
        raise ValueError(f"Não há iterações em comum para agregar {y_key}.")

    values = np.array([[run[step] for step in common_steps] for run in values_by_run])
    if smooth:
        values = np.apply_along_axis(_moving_average, 1, values)
    return np.array(common_steps), values.mean(axis=0), values.std(axis=0)


def _moving_average(values, window=100):
    series = pd.Series(values)
    if series.empty:
        return series.to_numpy()
    return series.rolling(
        window=window, min_periods=1
    ).mean().to_numpy()


def _plot_mean_std(ax, steps, mean, std, label, color=None):
    line, = ax.plot(steps, mean, label=f"{label} (média)", color=color)
    ax.fill_between(
        steps,
        mean - std,
        mean + std,
        color=line.get_color(),
        alpha=0.2,
        label=f"{label} (±1 desvio)",
    )
    return line


def _set_y_limits_from_iteration(ax, steps, mean_std_series, start_iter):
    if start_iter is None:
        return

    visible = np.asarray(steps) >= start_iter
    if not visible.any():
        return

    bounds = np.concatenate([
        values
        for mean, std in mean_std_series
        for values in (mean[visible] - std[visible], mean[visible] + std[visible])
    ])
    bounds = bounds[np.isfinite(bounds)]
    if not bounds.size:
        return

    low, high = bounds.min(), bounds.max()
    padding = max((high - low) * 0.05, abs(low) * 0.05, 0.05)
    ax.set_ylim(low - padding, high + padding)


def plot_loss(out_dir, seeds, type_model, smooth=False, start_iter=500):
    if type_model not in MODEL_TITLES:
        return
    runs = get_metrics_from_seed_logs(out_dir, seeds, type_model)
    t_steps, t_mean, t_std = _aggregate_metric(
        runs, "iteration", "train_loss", smooth=smooth
    )
    v_steps, v_mean, v_std = _aggregate_metric(
        runs, "iteration", "val_loss", smooth=smooth
    )

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    _plot_mean_std(ax, t_steps, t_mean, t_std, "treino")
    _plot_mean_std(ax, v_steps, v_mean, v_std, "validação")
    _set_y_limits_from_iteration(
        ax, t_steps, ((t_mean, t_std), (v_mean, v_std)), start_iter
    )

    plt.xlabel('iteração'); plt.ylabel('loss'); plt.legend()
    plt.title(MODEL_TITLES[type_model])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# Geração dos plots de loss, perplexity, gap e tempo
def plot_loss_perplexity_gap_time(
    out_dir, seeds, type_model, path_directory_save, name_file_save,
    smooth=False, start_iter=500
):
    runs = get_metrics_from_seed_logs(out_dir, seeds, type_model)
    metrics = {
        key: _aggregate_metric(runs, "iteration", key, smooth=smooth)
        for key in ("train_loss", "val_loss", "perplexity", "gap")
    }

    time_runs = []
    for run in runs:
        warmup_cutoff = sorted(run["time_iteration"])[4] if len(run["time_iteration"]) > 4 else None
        filtered = {
            "time_iteration": [],
            "time_per_iteration_ms": [],
        }
        for iteration, elapsed in zip(run["time_iteration"], run["time_per_iteration_ms"]):
            if warmup_cutoff is None or iteration > warmup_cutoff:
                filtered["time_iteration"].append(iteration)
                filtered["time_per_iteration_ms"].append(elapsed)
        time_runs.append(filtered)
    time_steps, time_mean, time_std = _aggregate_metric(
        time_runs, "time_iteration", "time_per_iteration_ms", smooth=smooth
    )

    steps, train_mean, train_std = metrics["train_loss"]
    _, val_mean, val_std = metrics["val_loss"]
    _, ppl_mean, ppl_std = metrics["perplexity"]
    _, gap_mean, gap_std = metrics["gap"]
    best = int(np.argmin(val_mean))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # 1) loss de treino e validação, com o melhor ponto marcado
    _plot_mean_std(ax[0, 0], steps, train_mean, train_std, 'treino')
    _plot_mean_std(ax[0, 0], steps, val_mean, val_std, 'validação')
    _set_y_limits_from_iteration(
        ax[0, 0], steps, ((train_mean, train_std), (val_mean, val_std)), start_iter
    )
    ax[0, 0].axvline(steps[best], color='gray', linestyle='--', label=f'melhor val (iter {steps[best]})')
    ax[0, 0].set_title('Loss'); ax[0, 0].set_xlabel('iteração'); ax[0, 0].set_ylabel('loss'); ax[0, 0].legend()

    # 2) perplexity de validação
    _plot_mean_std(ax[0, 1], steps, ppl_mean, ppl_std, 'perplexity', color='tab:orange')
    _set_y_limits_from_iteration(
        ax[0, 1], steps, ((ppl_mean, ppl_std),), start_iter
    )
    ax[0, 1].set_title('Perplexity de validação (exp da loss)'); ax[0, 1].set_xlabel('iteração'); ax[0, 1].set_ylabel('perplexity')

    # 3) diferença validação - treino (overfitting)
    _plot_mean_std(ax[1, 0], steps, gap_mean, gap_std, 'gap', color='tab:red')
    _set_y_limits_from_iteration(ax[1, 0], steps, ((gap_mean, gap_std),), start_iter)
    ax[1, 0].set_title('Gap de generalização (val - treino)'); ax[1, 0].set_xlabel('iteração'); ax[1, 0].set_ylabel('diferença de loss')

    # 4) tempo por iteração (ignorando as primeiras, que incluem aquecimento)
    _plot_mean_std(ax[1, 1], time_steps, time_mean, time_std, 'tempo', color='tab:green')
    _set_y_limits_from_iteration(
        ax[1, 1], time_steps, ((time_mean, time_std),), start_iter
    )
    ax[1, 1].set_title('Tempo por iteração'); ax[1, 1].set_xlabel('iteração'); ax[1, 1].set_ylabel('ms')

    fig.suptitle(MODEL_TITLES[type_model])
    plt.tight_layout()

    os.makedirs(path_directory_save, exist_ok=True)
    fig.savefig(name_file_save, dpi=150)
    plt.show()

    # --- resumo em texto, útil para a tabela do relatório ---
    print(f"Melhor val loss: {val_mean[best]:.4f} (iter {steps[best]}), perplexity: {ppl_mean[best]:.2f}")
    print(f"Loss final: treino {train_mean[-1]:.4f}, validação {val_mean[-1]:.4f}")
    print(f"Tempo por iteração (mediana): {np.median(time_mean):.1f} ms")


# Leitura do tempo de treinamento por época
def _as_paths(paths):
    if isinstance(paths, (str, os.PathLike)):
        return [paths]
    return list(paths)


def extract_epoch_time(log_files):
    return [
        elapsed
        for log_file in _as_paths(log_files)
        for elapsed in get_metrics_from_log(log_file)["time_per_iteration_ms"]
    ]


# Gera média e desvio padrão para tempos de treinamento
def generate_mean_std_time(log_files):
    it_time = extract_epoch_time(log_files)
    if not it_time:
        raise ValueError(f"Nenhum tempo por iteração encontrado nos arquivos {log_files}.")

    mean_time = sum(it_time) / len(it_time)
    std_time = (sum((x - mean_time) ** 2 for x in it_time) / len(it_time)) ** 0.5

    return mean_time, std_time


# Gera todas métricas para um modelo
def generate_all_metrics(benchmark_files, log_files):
    benchmarks = []
    for benchmark_file in _as_paths(benchmark_files):
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmarks.append(json.load(f))
    if not benchmarks:
        raise ValueError("É necessário fornecer ao menos um arquivo de benchmark.")

    sample_counts = [item["inference_num_samples"] for item in benchmarks]
    if any(count <= 0 for count in sample_counts):
        raise ValueError("Os benchmarks devem conter amostras de inferência.")
    total_samples = sum(sample_counts)
    mean_latency_s = sum(
        item["inference_latency_mean_s"] * count
        for item, count in zip(benchmarks, sample_counts)
    ) / total_samples
    variance_s = sum(
        count * (
            item["inference_latency_std_s"] ** 2
            + (item["inference_latency_mean_s"] - mean_latency_s) ** 2
        )
        for item, count in zip(benchmarks, sample_counts)
    ) / total_samples

    mean_time, std_time = generate_mean_std_time(log_files)
    return {
        "mean_latency_ms": mean_latency_s * 1000,
        "std_latency_ms": math.sqrt(variance_s) * 1000,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
    }


def fmt(mean, std):
    return f"{mean:.2f} ± {std:.2f}"


# Gera a tabela final para todos os modelos, com nome do modelo,
# tempo médio de treinamento, latência média de resposta e speedup
def generate_table_of_metrics_for_all_models(out_dir, seeds):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("É necessário fornecer ao menos uma seed para gerar a tabela.")

    model_types = [
        type_model
        for type_model in MODEL_TITLES
        if any(
            os.path.isdir(os.path.join(out_dir, f"{type_model}_s{seed}"))
            for seed in seeds
        )
    ]
    if not model_types:
        raise ValueError(f"Nenhum diretório de execução encontrado em {out_dir}.")

    model_names = {
        "seq": "Baseline",
        "inv": "Invertido",
        "par": "Paralelo",
        "par_shared": "Paralelo Compartilhado",
    }
    models = []
    for type_model in model_types:
        benchmark_files = [
            os.path.join(out_dir, f"{type_model}_s{seed}", "benchmark.json")
            for seed in seeds
        ]
        log_files = [
            os.path.join(out_dir, f"{type_model}_s{seed}", "log.txt")
            for seed in seeds
        ]
        models.append(generate_all_metrics(benchmark_files, log_files))

    base_latency = models[0]["mean_latency_ms"]

    final_metrics = {
        'Modelo': [model_names[type_model] for type_model in model_types],
        'Tempo Médio de época (s)\n± desvio padrão': [
            fmt(m['mean_time_ms'] / 1000, m['std_time_ms'] / 1000) for m in models
        ],
        'Latência Média (ms)\n± desvio padrão': [
            fmt(m['mean_latency_ms'], m['std_latency_ms']) for m in models
        ],
        'Speedup': [round(base_latency / m['mean_latency_ms'], 2) for m in models],
    }

    df = pd.DataFrame(final_metrics)
    styled_df = df.style.set_properties(**{
        'text-align': 'center',
        'white-space': 'pre-wrap'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('white-space', 'pre-wrap')]}
    ])

    return styled_df