#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
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
    'val_holdout_ppl_final',
    'train_time_s',
    'tokens_per_s',
    'mfu_mean_percent',
    'mfu_std_percent',
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
]

TRAINING_ENV_COLUMNS = ['world_size', 'gpu', 'peak_tflops']
TRAINING_QUALITY_METRICS = [
    'monitor_val_loss_final',
    'val_holdout_loss_final',
    'val_holdout_ppl_final',
    'epochs_trained',
]

@dataclass
class SummarizeConfig:
    out_root: str = str(OUTPUT_DIR / "c4")

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
    print(summary_table)

    table_csv_path = out_root / 'summary_table.csv'
    table_md_path = out_root / 'summary_table.md'
    summary_table.to_csv(table_csv_path)
    summary_table.to_markdown(table_md_path)

    logger.info("Tabelas salvas em:\n - %s\n - %s", table_csv_path, table_md_path)

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
            plt.plot(mean_metrics['iter'], mean_metrics['train_loss'], label=f"{variant} (Train)", alpha=0.3, linestyle='--')

        plt.title('Curvas de Loss (Treino e Monitoramento) por Variante')
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
