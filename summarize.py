#!/usr/bin/env python3

import json
import logging
import math
import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simple_parsing import ArgumentParser
from common import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

SUMMARY_METRICS = [
    'val_holdout_loss_final',
    'val_holdout_strided_loss_final',
    'val_holdout_ppl_final',
    'train_time_s',
    'tokens_per_s',
    'mfu_mean_percent',
    'mfu_std_percent',
    'peak_memory_mb',
    'inference_latency_mean_s',
    'inference_latency_std_s',
    'inference_tokens_per_sec',
]

INFERENCE_CONFIG_COLUMNS = [
    'inference_device',
    'inference_gpu',
    'inference_dtype',
    'inference_autocast_enabled',
    'inference_tf32_enabled',
    'inference_compile',
    'inference_max_new_tokens',
    'inference_prompt_length',
    'inference_num_samples',
    'inference_seed',
]

INFERENCE_METRICS = [
    'inference_latency_mean_s',
    'inference_latency_std_s',
    'inference_tokens_per_sec',
]

TRAINING_PERFORMANCE_METRICS = [
    'step_time_ms_mean',
    'step_time_ms_std',
    'tokens_per_s',
    'train_time_s',
    'mfu_mean_percent',
    'mfu_std_percent',
    'peak_memory_mb',
]

TRAINING_ENV_COLUMNS = ['world_size', 'gpu', 'peak_tflops']
TRAINING_QUALITY_METRICS = [
    'monitor_val_loss_final',
    'val_holdout_loss_final',
    'val_holdout_strided_loss_final',
    'val_holdout_ppl_final',
    'epochs_trained',
]

# Chaves do config.json que devem ser idênticas em todas as execuções agregadas.
# Espelha train.RESUME_LOCKED_KEYS, sem 'seed' e 'block_type', que variam por desenho.
CONSISTENCY_KEYS = [
    'batch_size', 'block_size', 'grad_accum', 'learning_rate', 'min_lr',
    'max_iters', 'lr_decay_iters', 'warmup_iters', 'weight_decay', 'beta1',
    'beta2', 'grad_clip', 'decay_lr', 'eval_iters', 'eval_interval',
    'n_layer', 'n_head', 'n_embd', 'bias', 'dropout',
    'dtype',
    'dataset', 'data_fingerprint',
]
LR_KEYS = ('learning_rate', 'min_lr')
# Diferenças de ambiente não impedem a agregação, mas são avisadas.
ENVIRONMENT_KEYS = ['python', 'torch', 'numpy', 'cuda', 'git_commit']

WINDOW_LOSSES_FILE = 'val_holdout_window_losses.npy'
POSITION_LOSSES_FILE = 'val_holdout_position_losses.npy'

# t de Student bicaudal a 95% para 1..30 graus de liberdade.
T_CRITICAL_95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]

@dataclass
class SummarizeConfig:
    out_root: str = str(OUTPUT_DIR / "c4")
    baseline: str = "seq"  # variante de referência da comparação pareada
    # Aceita learning_rate/min_lr diferentes entre variantes (iguais dentro de
    # cada uma), para o caso de LR escolhido por variante na varredura.
    allow_lr_per_variant: bool = False
    bootstrap_samples: int = 2000

def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"O arquivo {path} não contém um objeto JSON.")
    return data


def distinct_values(records: list[dict], columns: list[str]) -> dict[str, set]:
    """Retorna os valores não ausentes encontrados para cada campo."""
    values = {}
    for column in columns:
        found = set()
        for record in records:
            value = record.get(column)
            if value is not None and not pd.isna(value):
                try:
                    found.add(value)
                except TypeError:
                    found.add(repr(value))
        values[column] = found
    return values


def check_seeds(summaries: list[dict]) -> list[str]:
    """A comparação pareada exige as mesmas sementes em todas as variantes."""
    problems = []
    seeds: dict[str, list] = {}
    for summary in summaries:
        variant_seeds = seeds.setdefault(summary['block_type'], [])
        if summary.get('seed') in variant_seeds:
            problems.append(
                f"semente {summary.get('seed')} repetida na variante {summary['block_type']} "
                f"({Path(summary['_run_dir']).name})")
        variant_seeds.append(summary.get('seed'))
    seed_sets = {variant: sorted(values, key=str) for variant, values in seeds.items()}
    if len({tuple(values) for values in seed_sets.values()}) > 1:
        problems.append(f"sementes diferentes entre variantes: {seed_sets}")
    return problems


def check_consistency(summaries: list[dict], allow_lr_per_variant: bool) -> list[str]:
    """Confere, pelo config.json de cada execução, se a configuração e os dados
    são os mesmos. Guarda o ambiente lido em summary['_environment']."""
    problems = []
    reference: dict[tuple, tuple] = {}
    for summary in summaries:
        run_dir = Path(summary['_run_dir'])
        config_path = run_dir / 'config.json'
        if not config_path.is_file():
            problems.append(f"{run_dir.name}: config.json ausente; não é possível validar a execução")
            continue
        run_info = load_json(config_path)
        config = run_info.get('config') or {}
        summary['_environment'] = run_info.get('environment') or {}
        for key in CONSISTENCY_KEYS:
            scope = summary['block_type'] if (allow_lr_per_variant and key in LR_KEYS) else '*'
            value = json.dumps(config.get(key), sort_keys=True, default=str)
            ref_value, ref_name = reference.setdefault((scope, key), (value, run_dir.name))
            if value != ref_value:
                problems.append(f"{key}: {run_dir.name} ({value}) difere de {ref_name} ({ref_value})")
    return problems


def warn_environment(summaries: list[dict]) -> None:
    environments = [summary.get('_environment', {}) for summary in summaries]
    differing = [key for key, values in distinct_values(environments, ENVIRONMENT_KEYS).items()
                 if len(values) > 1]
    if differing:
        logger.warning("Ambientes de software diferentes entre as execuções: %s.", ', '.join(differing))
    if any(environment.get('git_dirty') for environment in environments):
        logger.warning("Há execuções feitas com alterações não commitadas no repositório (git_dirty).")


def holdout_loss(summary: dict) -> float | None:
    """Loss do holdout em nats; recuperada da perplexidade em resumos antigos."""
    loss = summary.get('val_holdout_loss_final')
    if loss is None and summary.get('val_holdout_ppl_final'):
        loss = math.log(summary['val_holdout_ppl_final'])
    return loss


def load_window_losses(summary: dict) -> np.ndarray | None:
    path = Path(summary['_run_dir']) / WINDOW_LOSSES_FILE
    return np.load(path).astype(np.float64) if path.is_file() else None


def t_interval(values: list[float]) -> tuple[float, float]:
    """Desvio padrão amostral e meia-largura do intervalo t de 95% da média."""
    n = len(values)
    sd = float(np.std(values, ddof=1))
    t_critical = T_CRITICAL_95[n - 2] if n - 1 <= len(T_CRITICAL_95) else 1.960
    return sd, t_critical * sd / math.sqrt(n)


def position_table(summaries: list[dict], baseline: str) -> pd.DataFrame | None:
    """Loss média do holdout por posição na janela, por variante, em faixas que
    dobram de tamanho (0, 1, 2-3, 4-7, ...). O token previsto na posição p é
    condicionado a p + 1 tokens, então as primeiras faixas mostram o
    comportamento com pouco contexto e a última, com mais de meia janela."""
    by_variant: dict[str, list] = {}
    for summary in summaries:
        path = Path(summary['_run_dir']) / POSITION_LOSSES_FILE
        if not path.is_file():
            logger.warning(
                "Loss por posição ausente em %s; análise por posição omitida. Para gerá-la sem "
                "retreinar: python train.py ... --init_from resume --eval_only.",
                Path(summary['_run_dir']).name)
            return None
        by_variant.setdefault(summary['block_type'], []).append(np.load(path).astype(np.float64))
    sizes = {len(losses) for runs in by_variant.values() for losses in runs}
    if len(sizes) != 1:
        return None
    size = sizes.pop()
    means = {variant: np.mean(runs, axis=0) for variant, runs in by_variant.items()}
    edges = [0, 1]
    while edges[-1] < size:
        edges.append(min(edges[-1] * 2, size))
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        row = {'positions': str(low) if high - low == 1 else f'{low}-{high - 1}'}
        for variant in sorted(means):
            row[f'loss_{variant}'] = float(means[variant][low:high].mean())
        if baseline in means:
            for variant in sorted(means):
                if variant != baseline:
                    row[f'diff_{variant}'] = row[f'loss_{variant}'] - row[f'loss_{baseline}']
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(summaries: list[dict], baseline: str, bootstrap_samples: int) -> pd.DataFrame | None:
    """Diferença pareada por semente da loss do holdout (variante - baseline), em nats.

    Dois intervalos de 95%, que medem incertezas diferentes:
    - ci95_seeds: t de Student sobre as diferenças por semente (variância entre
      sementes: inicialização e ordem dos dados).
    - ci95_windows: bootstrap pareado por janela do holdout, com os modelos
      treinados fixos (erro amostral do conjunto de avaliação). Reamostra
      janelas independentes; documentos que cruzam janelas vizinhas não são
      modelados.
    """
    runs = {(summary['block_type'], summary.get('seed')): summary for summary in summaries}
    variants = sorted({variant for variant, _ in runs})
    if baseline not in variants:
        logger.warning("Variante de referência '%s' ausente; comparação pareada omitida.", baseline)
        return None
    seeds = sorted((seed for variant, seed in runs if variant == baseline), key=str)

    rows = []
    for variant in variants:
        if variant == baseline:
            continue
        diffs, window_diffs, strided_diffs = [], [], []
        for seed in seeds:
            run, reference = runs[(variant, seed)], runs[(baseline, seed)]
            loss, reference_loss = holdout_loss(run), holdout_loss(reference)
            if loss is None or reference_loss is None:
                continue
            diffs.append(loss - reference_loss)
            strided = run.get('val_holdout_strided_loss_final')
            reference_strided = reference.get('val_holdout_strided_loss_final')
            if strided is not None and reference_strided is not None:
                strided_diffs.append(strided - reference_strided)
            windows, reference_windows = load_window_losses(run), load_window_losses(reference)
            if (windows is not None and reference_windows is not None
                    and windows.shape == reference_windows.shape):
                window_diffs.append(windows - reference_windows)
        n = len(diffs)
        if n == 0:
            continue
        mean_diff = float(np.mean(diffs))
        row = {
            'variant': variant, 'baseline': baseline, 'n_seeds': n,
            'mean_diff_nats': mean_diff, 'ppl_ratio': math.exp(mean_diff),
            'sd_diff_nats': None, 'ci95_seeds_low': None, 'ci95_seeds_high': None,
            'ci95_windows_low': None, 'ci95_windows_high': None,
            # Mesma diferença na métrica secundária (janela deslizante de meio bloco).
            'strided_mean_diff_nats': None,
            'strided_ci95_seeds_low': None, 'strided_ci95_seeds_high': None,
        }
        if n >= 2:
            sd, half = t_interval(diffs)
            row.update(sd_diff_nats=sd, ci95_seeds_low=mean_diff - half, ci95_seeds_high=mean_diff + half)
        if len(strided_diffs) == n:
            strided_mean = float(np.mean(strided_diffs))
            row.update(strided_mean_diff_nats=strided_mean)
            if n >= 2:
                _, half = t_interval(strided_diffs)
                row.update(strided_ci95_seeds_low=strided_mean - half,
                           strided_ci95_seeds_high=strided_mean + half)
        if window_diffs and len(window_diffs) == n:
            mean_window_diff = np.mean(window_diffs, axis=0)
            rng = np.random.default_rng(0)
            size = len(mean_window_diff)
            boot = np.array([mean_window_diff[rng.integers(0, size, size)].mean()
                             for _ in range(bootstrap_samples)])
            low, high = np.percentile(boot, [2.5, 97.5])
            row.update(ci95_windows_low=float(low), ci95_windows_high=float(high))
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def main():
    parser = ArgumentParser(description="Agrega resultados e gera tabelas/gráficos do treinamento.")
    parser.add_arguments(SummarizeConfig, dest="cfg")
    args = parser.parse_args()
    cfg: SummarizeConfig = args.cfg

    out_root = Path(cfg.out_root)
    if not out_root.is_dir():
        logger.error("Diretório %s não encontrado ou não é um diretório.", out_root)
        return

    summaries = []
    metrics_dict = {}
    benchmark_run_count = 0

    for run_dir in sorted(out_root.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / 'summary.json'
        if not summary_path.is_file():
            continue

        summary = load_json(summary_path)
        variant = summary.get('block_type')
        if not variant:
            logger.warning("Ignorando %s: campo 'block_type' ausente.", summary_path)
            continue

        bench_path = run_dir / 'benchmark.json'
        if bench_path.is_file():
            benchmark_data = load_json(bench_path)
            summary.update(benchmark_data)
            benchmark_run_count += 1

        summary['_run_dir'] = str(run_dir)
        summaries.append(summary)

        metrics_path = run_dir / f'metrics_{variant}.csv'
        if metrics_path.is_file():
            df_metrics = pd.read_csv(metrics_path)
            required_columns = {'iter', 'train_loss', 'val_loss'}
            if required_columns.issubset(df_metrics.columns):
                df_metrics = df_metrics.copy()
                for column in ('iter', 'train_loss', 'val_loss'):
                    df_metrics[column] = pd.to_numeric(
                        df_metrics[column], errors='coerce'
                    )
                df_metrics = df_metrics.dropna(subset=['iter'])
                metrics_dict.setdefault(variant, []).append(df_metrics)
            else:
                faltantes = sorted(required_columns - set(df_metrics.columns))
                logger.warning("Ignorando curvas de %s: faltam colunas %s.", metrics_path, faltantes)

    if not summaries:
        logger.error("Nenhum arquivo summary.json válido encontrado.")
        return

    problems = check_seeds(summaries) + check_consistency(summaries, cfg.allow_lr_per_variant)
    if problems:
        logger.error("Agregação recusada: as execuções não são comparáveis.")
        for problem in problems:
            logger.error(" - %s", problem)
        sys.exit(1)
    warn_environment(summaries)

    for summary in summaries:
        summary['val_holdout_loss_final'] = holdout_loss(summary)

    benchmark_summaries = [
        summary for summary in summaries
        if 'inference_latency_mean_s' in summary
    ]
    inference_values = distinct_values(benchmark_summaries, INFERENCE_CONFIG_COLUMNS)
    inconsistent_inference = [column for column, values in inference_values.items() if len(values) > 1]
    incomplete_inference = [
        column for column in INFERENCE_CONFIG_COLUMNS
        if benchmark_summaries and any(column not in summary for summary in benchmark_summaries)
    ]
    inconsistent_inference.extend(column for column in incomplete_inference if column not in inconsistent_inference)
    
    if inconsistent_inference:
        logger.warning(
            "Configurações de benchmark diferentes entre as execuções: %s. As métricas de inferência serão omitidas.",
            ', '.join(inconsistent_inference)
        )


        for summary in summaries:
            for metric in INFERENCE_METRICS:
                summary[metric] = None

    if benchmark_run_count < len(summaries):
        logger.warning(
            "Benchmark disponível em %s de %s execuções; as médias de inferência "
            "considerarão somente as execuções que possuem benchmark.json.",
            benchmark_run_count, len(summaries)
        )

    dataset_values = distinct_values(summaries, ['dataset']).get('dataset', set())
    if len(dataset_values) > 1:
        logger.warning(
            "Foram encontrados datasets diferentes. Todas as métricas de "
            "treinamento serão omitidas da tabela consolidada."
        )
        for summary in summaries:
            for metric in TRAINING_QUALITY_METRICS + TRAINING_PERFORMANCE_METRICS:
                summary[metric] = None

    training_env_values = distinct_values(summaries, TRAINING_ENV_COLUMNS)
    inconsistent_training_env = [
        column for column, values in training_env_values.items() if len(values) > 1
    ]
    if inconsistent_training_env:
        logger.warning(
            "Ambientes de treinamento diferentes entre as execuções: %s. As métricas de desempenho "
            "de treinamento serão omitidas da tabela consolidada; métricas de perda e épocas "
            "continuarão disponíveis.", 
            ', '.join(inconsistent_training_env)
        )

        for summary in summaries:
            for metric in TRAINING_PERFORMANCE_METRICS:
                summary[metric] = None

    df = pd.DataFrame(summaries)
    agg_cols = [column for column in SUMMARY_METRICS if column in df.columns]
    if not agg_cols:
        logger.error("Os arquivos summary.json não contêm métricas reconhecidas para agregar.")
        return

    summary_table = (
        df.groupby('block_type')[agg_cols]
        .agg(['mean', 'std'])
        .round(4)
    )

    logger.info("=== Resumo consolidado (média e desvio padrão) ===")
    logger.info("Execuções por variante: %s", df.groupby('block_type').size().to_dict())
    print(summary_table)

    table_csv_path = out_root / 'summary_table.csv'
    table_md_path = out_root / 'summary_table.md'
    summary_table.to_csv(table_csv_path)
    summary_table.to_markdown(table_md_path)

    logger.info("Tabelas salvas em:\n - %s\n - %s", table_csv_path, table_md_path)

    paired = paired_comparison(summaries, cfg.baseline, cfg.bootstrap_samples)
    if paired is not None:
        logger.info("=== Comparação pareada por semente: loss do holdout em nats (variante - %s) ===",
                    cfg.baseline)
        print(paired.round(5).to_string(index=False))
        paired_csv_path = out_root / 'paired_comparison.csv'
        paired_md_path = out_root / 'paired_comparison.md'
        paired.to_csv(paired_csv_path, index=False)
        paired.to_markdown(paired_md_path, index=False)
        logger.info("Comparação pareada salva em:\n - %s\n - %s", paired_csv_path, paired_md_path)
        for row in paired.to_dict('records'):
            low, high = row['ci95_seeds_low'], row['ci95_seeds_high']
            inconclusive = low is None or pd.isna(low) or (low <= 0.0 <= high)
            if inconclusive and row['n_seeds'] < 5:
                logger.warning(
                    "%s vs %s: com %d semente(s) o intervalo de 95%% inclui zero ou não pôde ser "
                    "calculado. Se a diferença importar, amplie para 5 ou mais sementes "
                    "(ex.: SEEDS=\"1337 1338 1339 1340 1341\" ./run_all.sh).",
                    row['variant'], row['baseline'], row['n_seeds'])

    # o ranking das variantes deve ser o mesmo com janelas disjuntas (métrica primária) e com janela deslizante.
    ranking_columns = ['val_holdout_loss_final', 'val_holdout_strided_loss_final']
    if all(column in df.columns and df[column].notna().all() for column in ranking_columns):
        means = df.groupby('block_type')[ranking_columns].mean()
        primary_order = list(means[ranking_columns[0]].sort_values().index)
        strided_order = list(means[ranking_columns[1]].sort_values().index)
        if primary_order == strided_order:
            logger.info("Ranking das variantes igual nas duas métricas do holdout "
                        "(janelas disjuntas e janela deslizante): %s", ' < '.join(primary_order))
        else:
            logger.warning("Ranking das variantes difere entre as métricas do holdout: janelas "
                           "disjuntas %s; janela deslizante %s. A conclusão depende do protocolo.",
                           ' < '.join(primary_order), ' < '.join(strided_order))

    positions = position_table(summaries, cfg.baseline)
    if positions is not None:
        logger.info("=== Loss do holdout por posição na janela (média das sementes) ===")
        print(positions.round(4).to_string(index=False))
        positions_csv_path = out_root / 'position_loss.csv'
        positions.to_csv(positions_csv_path, index=False)

        loss_columns = [column for column in positions.columns if column.startswith('loss_')]
        diff_columns = [column for column in positions.columns if column.startswith('diff_')]
        figure, axes = plt.subplots(1, 2 if diff_columns else 1, figsize=(14, 6), squeeze=False)
        x = range(len(positions))
        for column in loss_columns:
            axes[0][0].plot(x, positions[column], marker='o', label=column[len('loss_'):])
        axes[0][0].set_title('Loss por posição na janela')
        axes[0][0].set_ylabel('Loss (nats)')
        if diff_columns:
            for column in diff_columns:
                axes[0][1].plot(x, positions[column], marker='o', label=column[len('diff_'):])
            axes[0][1].axhline(0.0, color='black', linewidth=1)
            axes[0][1].set_title(f'Diferença para {cfg.baseline} por posição')
            axes[0][1].set_ylabel(f'Loss da variante - loss de {cfg.baseline} (nats)')
        for axis in axes[0]:
            axis.set_xticks(list(x))
            axis.set_xticklabels(positions['positions'], rotation=45, ha='right')
            axis.set_xlabel('Posição na janela (faixas)')
            axis.legend()
            axis.grid(True, linestyle=':', alpha=0.7)
        positions_plot_path = out_root / 'position_loss.png'
        figure.savefig(positions_plot_path, dpi=300, bbox_inches='tight')
        plt.close(figure)
        logger.info("Loss por posição salva em:\n - %s\n - %s", positions_csv_path, positions_plot_path)

    if metrics_dict:
        plt.figure(figsize=(12, 7))

        for variant, dataframes in sorted(metrics_dict.items()):
            all_metrics = pd.concat(dataframes, ignore_index=True)
            mean_metrics = (
                all_metrics.groupby('iter')[['train_loss', 'val_loss']]
                .mean()
                .reset_index()
            )

            plt.plot(mean_metrics['iter'], mean_metrics['val_loss'], label=f"{variant} (Val. monitoramento)", linewidth=2)
            plt.plot(mean_metrics['iter'], mean_metrics['train_loss'], label=f"{variant} (amostra fixa do treino)", alpha=0.3, linestyle='--')

        plt.title('Curvas de loss (treino e monitoramento) por variante')
        plt.xlabel('Iteração')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)

        plot_out_path = out_root / 'loss_curves.png'
        plt.savefig(plot_out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Gráfico de perdas salvo em: %s", plot_out_path)

if __name__ == '__main__':
    main()
