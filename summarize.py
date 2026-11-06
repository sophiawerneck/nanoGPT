#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Agrega resultados e gera tabelas/gráficos do treinamento.")
    parser.add_argument('--out_root', type=str, required=True, help="Diretório raiz das saídas (ex: out/c4)")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    if not out_root.exists():
        print(f"Erro: Diretório {out_root} não encontrado.")
        return

    summaries = []
    metrics_dict = {}

    for run_dir in out_root.iterdir():
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
            # Caso benchmark.py gerem JSONs próprios, você pode carregá-los aqui
            # e dar um update() no dicionário summary antes do append.
            summaries.append(summary)

        variant = summary.get("block_type")
        metrics_path = run_dir / f"metrics_{variant}.csv"
        
        if metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            metrics_dict.setdefault(variant, []).append(df_metrics)

    if not summaries:
        print("Nenhum arquivo summary.json encontrado. O treinamento finalizou corretamente?")
        return

    df = pd.DataFrame(summaries)
    
    agg_cols = ['monitor_val_loss_final', 'step_time_ms_mean', 'tokens_per_s', 'train_time_s']
    
    valid_cols = [c for c in agg_cols if c in df.columns]
    
    summary_table = df.groupby('block_type')[valid_cols].agg(['mean', 'std']).round(4)
    
    print("\n=== Resumo consolidado (média e desvio padrão) ===")
    print(summary_table)
    
    table_csv_path = out_root / "summary_table.csv"
    table_md_path = out_root / "summary_table.md"
    summary_table.to_csv(table_csv_path)
    summary_table.to_markdown(table_md_path)
    
    print(f"\nTabelas salvas em:\n - {table_csv_path}\n - {table_md_path}")

    if metrics_dict:
        plt.figure(figsize=(12, 7))
        
        for variant, dfs in metrics_dict.items():
            all_metrics = pd.concat(dfs)
            mean_metrics = all_metrics.groupby('iter').mean().reset_index()
            
            plt.plot(mean_metrics['iter'], mean_metrics['val_loss'], 
                     label=f"{variant} (Val)", linewidth=2)
            plt.plot(mean_metrics['iter'], mean_metrics['train_loss'], 
                     label=f"{variant} (Train)", alpha=0.3, linestyle='--')

        plt.title('Curvas de Loss (Treino e Validação) por Variante (Média das Sementes)')
        plt.xlabel('Iteração')
        plt.ylabel('Loss (Cross Entropy)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plot_out_path = out_root / "loss_curves.png"
        plt.savefig(plot_out_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de perdas salvo em: {plot_out_path}")

if __name__ == "__main__":
    main()