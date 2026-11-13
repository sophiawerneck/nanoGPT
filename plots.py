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

    numeric = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))'
    holdout_loss_pattern = re.compile(
        r'step\s+(\d+):\s*train loss\s+' + numeric
        + r'.*?\bval holdout loss\s+' + numeric
    )
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
            match = holdout_loss_pattern.search(line) or loss_pattern.search(line)
            if match:
                iteration, train_loss, val_loss = match.groups()
                train_loss, val_loss = float(train_loss), float(val_loss)
                metrics["iteration"].append(int(iteration))
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["perplexity"].append(float(np.exp(val_loss)))
                metrics["gap"].append(val_loss - train_loss)
                continue

            match = time_pattern.search(line)
            if match and "(aquecimento)" not in line:
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


def _aggregate_metric(runs, x_key, y_key, smooth=False, ddof=0):
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
    std = (
        values.std(axis=0, ddof=ddof)
        if values.shape[0] > ddof
        else np.full(values.shape[1], np.nan)
    )
    return (
        np.array(common_steps),
        values.mean(axis=0),
        std,
    )


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
    smooth=False, start_iter=1000
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("É necessário fornecer ao menos uma seed.")
    if type_model not in MODEL_TITLES:
        raise ValueError(f"Tipo de bloco desconhecido: {type_model}.")

    model_names = {
        "seq": "Baseline",
        "inv": "Invertido",
        "par": "Paralelo Independente",
        "par_shared": "Paralelo Compartilhado",
    }
    aggregated = {}
    for model_type in MODEL_TITLES:
        run_dirs = [
            os.path.join(out_dir, f"{model_type}_s{seed}")
            for seed in seeds
            if os.path.isdir(os.path.join(out_dir, f"{model_type}_s{seed}"))
        ]
        if not run_dirs:
            continue

        runs = [
            get_metrics_from_log(os.path.join(run_dir, "log.txt"))
            for run_dir in run_dirs
        ]
        perplexity_runs = []
        for run in runs:
            run_with_perplexity = dict(run)
            run_with_perplexity["perplexity"] = np.exp(
                np.asarray(run["val_loss"], dtype=float)
            ).tolist()
            perplexity_runs.append(run_with_perplexity)

        metrics = {
            "val_loss": _aggregate_metric(
                runs, "iteration", "val_loss", ddof=1
            ),
            "perplexity": _aggregate_metric(
                perplexity_runs, "iteration", "perplexity", ddof=1
            ),
            "gap": _aggregate_metric(
                runs, "iteration", "gap", ddof=1
            ),
        }

        time_runs = []
        for run in runs:
            timed_iterations = sorted(run["time_iteration"])
            warmup_cutoff = timed_iterations[4] if len(timed_iterations) > 4 else None
            filtered = {
                "time_iteration": [],
                "time_per_iteration_ms": [],
            }
            for iteration, elapsed in zip(
                run["time_iteration"], run["time_per_iteration_ms"]
            ):
                if warmup_cutoff is None or iteration > warmup_cutoff:
                    filtered["time_iteration"].append(iteration)
                    filtered["time_per_iteration_ms"].append(elapsed)
            time_runs.append(filtered)
        metrics["time"] = _aggregate_metric(
            time_runs, "time_iteration", "time_per_iteration_ms",
            smooth=smooth, ddof=1
        )
        aggregated[model_type] = metrics

    if not aggregated:
        raise ValueError(f"Nenhum diretório de execução encontrado em {out_dir}.")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.get_cmap("tab10").colors
    metric_axes = (
        (ax[0, 0], "val_loss", "Loss de Validação", "Loss de Validação"),
        (ax[0, 1], "perplexity", "Perplexidade de Validação", "Perplexidade"),
        (ax[1, 0], "gap", "Gap de Generalização (Val - Treino)", "diferença de loss"),
        (ax[1, 1], "time", "Tempo por Iteração", "ms"),
    )

    def plot_model_series(axis, steps, mean, std, label, color):
        line, = axis.plot(steps, mean, label=label, color=color)
        axis.fill_between(
            steps, mean - std, mean + std, color=line.get_color(), alpha=0.2
        )

    for index, (model_type, metrics) in enumerate(aggregated.items()):
        color = colors[index % len(colors)]
        label = model_names[model_type]
        for axis, metric_key, _, _ in metric_axes:
            steps, mean, std = metrics[metric_key]
            if metric_key in ("val_loss", "perplexity"):
                visible = (steps >= start_iter) & (steps <= 6000)
                steps, mean, std = steps[visible], mean[visible], std[visible]
            elif metric_key == "gap":
                visible = (steps >= 0) & (steps <= 6000)
                steps, mean, std = steps[visible], mean[visible], std[visible]
            plot_model_series(axis, steps, mean, std, label, color)

    for axis, metric_key, title, ylabel in metric_axes:
        axis.set_title(title)
        axis.set_xlabel("passo")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", alpha=0.6)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, 0.01), ncol=len(labels)
    )

    def apply_limits(axis, metric_key, min_step, max_step):
        bounds = []
        for metrics in aggregated.values():
            steps, mean, std = metrics[metric_key]
            visible = (steps >= min_step) & (steps <= max_step)
            if visible.any():
                visible_mean = mean[visible]
                visible_std = std[visible]
                lower = np.where(
                    np.isfinite(visible_std), visible_mean - visible_std, visible_mean
                )
                upper = np.where(
                    np.isfinite(visible_std), visible_mean + visible_std, visible_mean
                )
                bounds.extend((
                    lower,
                    upper,
                ))
        axis.set_xlim(left=min_step, right=max_step)
        if not bounds:
            return
        values = np.concatenate(bounds)
        values = values[np.isfinite(values)]
        if not values.size:
            return
        low, high = values.min(), values.max()
        padding = (high - low) * 0.05
        if padding == 0:
            padding = max(abs(low) * 0.05, 0.05)
        axis.set_ylim(low - padding, high + padding)

    zoom_start_step = 1000 if start_iter is None else start_iter
    apply_limits(ax[0, 0], "val_loss", zoom_start_step, 6000)
    apply_limits(ax[0, 1], "perplexity", zoom_start_step, 6000)
    apply_limits(ax[1, 0], "gap", 0, 6000)

    fig.suptitle("Comparação dos Modelos")
    plt.tight_layout(rect=(0, 0.07, 1, 0.95))

    os.makedirs(path_directory_save, exist_ok=True)
    fig.savefig(name_file_save, dpi=300, bbox_inches="tight")
    plt.show()

    for model_type, metrics in aggregated.items():
        val_steps, val_mean, _ = metrics["val_loss"]
        ppl_steps, ppl_mean, _ = metrics["perplexity"]
        _, time_mean, _ = metrics["time"]
        val_regular = val_steps <= 6000
        ppl_regular = ppl_steps <= 6000
        final_step = min(val_steps[val_regular][-1], ppl_steps[ppl_regular][-1])
        print(
            f"{model_names[model_type]} — passo final {final_step}: "
            f"loss de validação {val_mean[val_regular][-1]:.4f}, "
            f"perplexidade {ppl_mean[ppl_regular][-1]:.2f}; "
            f"tempo mediano {np.median(time_mean):.1f} ms"
        )


def plot_comparative_metrics_for_all_models(
    out_dir, seeds, path_directory_save=None
):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("É necessário fornecer ao menos uma seed para gerar os gráficos.")

    model_names = {
        "seq": "Baseline",
        "inv": "Invertido",
        "par": "Paralelo",
        "par_shared": "Paralelo Compartilhado",
    }
    model_runs = {}
    for type_model in MODEL_TITLES:
        run_dirs = [
            os.path.join(out_dir, f"{type_model}_s{seed}")
            for seed in seeds
            if os.path.isdir(os.path.join(out_dir, f"{type_model}_s{seed}"))
        ]
        if run_dirs:
            model_runs[type_model] = [
                get_metrics_from_log(os.path.join(run_dir, "log.txt"))
                for run_dir in run_dirs
            ]
    if not model_runs:
        raise ValueError(f"Nenhum diretório de execução encontrado em {out_dir}.")

    aggregated = {}
    for type_model, runs in model_runs.items():
        metrics = {
            key: _aggregate_metric(runs, "iteration", key)
            for key in ("val_loss", "perplexity", "gap")
        }

        time_runs = []
        for run in runs:
            timed_iterations = sorted(run["time_iteration"])
            warmup_cutoff = timed_iterations[4] if len(timed_iterations) > 4 else None
            filtered = {
                "time_iteration": [],
                "time_per_iteration_ms": [],
            }
            for iteration, elapsed in zip(
                run["time_iteration"], run["time_per_iteration_ms"]
            ):
                if warmup_cutoff is None or iteration > warmup_cutoff:
                    filtered["time_iteration"].append(iteration)
                    filtered["time_per_iteration_ms"].append(elapsed)
            filtered["time_per_iteration_ms"] = pd.Series(
                filtered["time_per_iteration_ms"]
            ).rolling(window=5, min_periods=1, center=True).mean().to_list()
            time_runs.append(filtered)
        metrics["time"] = _aggregate_metric(
            time_runs, "time_iteration", "time_per_iteration_ms"
        )
        aggregated[type_model] = metrics

    if path_directory_save is None:
        path_directory_save = os.path.join(out_dir, "plots_comparativos")
    os.makedirs(path_directory_save, exist_ok=True)

    colors = plt.get_cmap("tab10").colors
    paths = {}

    def plot_mean(ax, steps, mean, label, color, linestyle="-"):
        ax.plot(steps, mean, label=label, color=color, linestyle=linestyle)

    def plot_series(ax, steps, mean, std, label, color):
        plot_mean(ax, steps, mean, label, color)
        ax.fill_between(
            steps, mean - std, mean + std, color=color, alpha=0.2
        )

    def apply_zoom(ax, metric_key, start_step):
        bounds = []
        for metrics in aggregated.values():
            steps, mean, std = metrics[metric_key]
            visible = steps >= start_step
            if visible.any():
                bounds.extend((mean[visible] - std[visible], mean[visible] + std[visible]))
        if not bounds:
            return

        values = np.concatenate(bounds)
        values = values[np.isfinite(values)]
        if not values.size:
            return
        low, high = values.min(), values.max()
        padding = max((high - low) * 0.05, max(abs(low), abs(high), 1) * 0.01)
        ax.set_xlim(left=start_step)
        ax.set_ylim(low - padding, high + padding)

    zoom_start_step = 250
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (type_model, metrics) in enumerate(aggregated.items()):
        color = colors[index % len(colors)]
        steps, mean, std = metrics["val_loss"]
        plot_series(
            ax, steps, mean, std, model_names[type_model], color
        )
    ax.set(
        title="Loss de Validação por Modelo",
        xlabel="Passo",
        ylabel="Loss de Validação",
    )
    apply_zoom(ax, "val_loss", zoom_start_step)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    paths["loss"] = os.path.join(path_directory_save, "loss_comparativo.png")
    fig.savefig(paths["loss"], dpi=300)
    plt.close(fig)

    plot_specs = (
        ("perplexity", "Perplexidade de Validação", "Perplexidade de Validação",
         "perplexidade_validacao_comparativo.png"),
        ("gap", "Gap de Generalização (Val - Treino)",
         "Gap de Generalização (Val - Treino)",
         "gap_generalizacao_comparativo.png"),
        ("time", "Tempo por iteração", "Tempo por iteração (ms)",
         "tempo_por_iteracao_comparativo.png"),
    )
    for metric_key, title, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, (type_model, metrics) in enumerate(aggregated.items()):
            steps, mean, std = metrics[metric_key]
            color = colors[index % len(colors)]
            plot_series(ax, steps, mean, std, model_names[type_model], color)
        if metric_key == "perplexity":
            apply_zoom(ax, metric_key, zoom_start_step)
        ax.set(title=title, xlabel="Passo", ylabel=ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        paths[metric_key] = os.path.join(path_directory_save, filename)
        fig.savefig(paths[metric_key], dpi=300)
        plt.close(fig)

    return paths


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


def extract_benchmark_latencies(bench_log_files):
    latency_pattern = re.compile(
        r'\b\d+/\d+\s+time:\s*'
        r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)s\b'
    )
    latencies_ms = []
    for bench_log_file in _as_paths(bench_log_files):
        with open(bench_log_file, encoding="utf-8") as bench_log:
            for line in bench_log:
                match = latency_pattern.search(line)
                if match:
                    latencies_ms.append(float(match.group(1)) * 1000)
    return latencies_ms


# Gera todas métricas para um modelo
def generate_all_metrics(benchmark_files, log_files, bench_log_files=None):
    benchmark_paths = _as_paths(benchmark_files)
    if bench_log_files is None and benchmark_paths:
        candidate_bench_logs = [
            os.path.join(os.path.dirname(path), "bench.log")
            for path in benchmark_paths
        ]
        if all(os.path.isfile(path) for path in candidate_bench_logs):
            bench_log_files = candidate_bench_logs

    if bench_log_files is not None:
        latencies_ms = extract_benchmark_latencies(bench_log_files)
        if not latencies_ms:
            raise ValueError("Nenhuma amostra de latência encontrada nos arquivos bench.log.")
        mean_latency_ms = float(np.mean(latencies_ms))
        std_latency_ms = float(np.std(latencies_ms, ddof=0))
        median_latency_ms = float(np.median(latencies_ms))
        iqr_latency_ms = float(
            np.percentile(latencies_ms, 75) - np.percentile(latencies_ms, 25)
        )
    else:
        benchmarks = []
        for benchmark_file in benchmark_paths:
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
        mean_latency_ms = mean_latency_s * 1000
        std_latency_ms = math.sqrt(variance_s) * 1000
        # Summary JSON files do not retain individual samples; these fallback
        # quantiles describe the per-seed means, not the underlying samples.
        seed_means_ms = [
            item["inference_latency_mean_s"] * 1000 for item in benchmarks
        ]
        median_latency_ms = float(np.median(seed_means_ms))
        iqr_latency_ms = float(
            np.percentile(seed_means_ms, 75) - np.percentile(seed_means_ms, 25)
        )

    mean_time, std_time = generate_mean_std_time(log_files)
    return {
        "mean_latency_ms": mean_latency_ms,
        "std_latency_ms": std_latency_ms,
        "median_latency_ms": median_latency_ms,
        "iqr_latency_ms": iqr_latency_ms,
        "mean_time_ms": mean_time,
        "std_time_ms": std_time,
    }


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


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
        run_dirs = [
            os.path.join(out_dir, f"{type_model}_s{seed}")
            for seed in seeds
            if os.path.isdir(os.path.join(out_dir, f"{type_model}_s{seed}"))
        ]
        benchmark_files = [
            os.path.join(run_dir, "benchmark.json")
            for run_dir in run_dirs
        ]
        bench_log_files = [
            os.path.join(run_dir, "bench.log")
            for run_dir in run_dirs
        ]
        log_files = [
            os.path.join(run_dir, "log.txt")
            for run_dir in run_dirs
        ]
        models.append(generate_all_metrics(
            benchmark_files, log_files, bench_log_files=bench_log_files
        ))

    base_latency = models[0]["mean_latency_ms"]
    base_median_latency = models[0]["median_latency_ms"]

    final_metrics = {
        'Modelo': [model_names[type_model] for type_model in model_types],
        'Tempo Médio de época (s)\n± desvio padrão': [
            fmt(m['mean_time_ms'] / 1000, m['std_time_ms'] / 1000) for m in models
        ],
        'Latência Média (ms)\n± desvio padrão': [
            f"{m['mean_latency_ms']:.2f} ± {m['std_latency_ms']:.4f}"
            for m in models
        ],
        'Latência Mediana (ms)\n(IQR)': [
            f"{m['median_latency_ms']:.2f} ({m['iqr_latency_ms']:.4f})"
            for m in models
        ],
        'Speedup (latência média)': [
            round(base_latency / m['mean_latency_ms'], 2) for m in models
        ],
        'Speedup (latência mediana)': [
            round(base_median_latency / m['median_latency_ms'], 2)
            for m in models
        ],
    }

    df = pd.DataFrame(final_metrics)
    styled_df = df.style.set_properties(**{
        'text-align': 'center',
        'white-space': 'pre-wrap'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center'), ('white-space', 'pre-wrap')]}
    ])

    return styled_df