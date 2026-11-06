#!/usr/bin/env bash
# Script principal de execução: dados -> treino de todas as variantes e sementes.
# Idempotente: execuções concluídas são puladas e as interrompidas são retomadas.
#
# As etapas de avaliação (Membro 4), benchmark e agregação (Membro 3) rodam
# automaticamente quando os respectivos scripts existirem no repositório:
#   evaluate.py  --out_dir <execução>   -> perda/perplexidade final
#   benchmark.py --out_dir <execução>   -> latência de inferência
#   summarize.py                        -> tabelas e speedup
#
# Variáveis de ambiente (opcionais):
#   SEEDS="1337 1338 1339"   sementes (padrão: 3)
#   VARIANTS="seq inv par"   variantes
#   NPROC=1                  GPUs por execução (usa torchrun se > 1)
#   TRAIN_ARGS=""            argumentos extras para train.py (ex.: --peak_tflops 989)
set -euo pipefail
cd "$(dirname "$0")"

SEEDS="${SEEDS:-1337 1338 1339}"
VARIANTS="${VARIANTS:-seq inv par}"
NPROC="${NPROC:-1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

python prepare_c4.py

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    out="out/${variant}_s${seed}"

    if [[ -f "$out/summary.json" ]]; then
      echo ">> $out já treinado."
    else
      init="scratch"
      [[ -f "$out/ckpt.pt" ]] && init="resume"
      echo ">> Treinando $variant (semente $seed, $init)"
      if [[ "$NPROC" -gt 1 ]]; then
        torchrun --standalone --nproc_per_node="$NPROC" train.py \
          --block_type "$variant" --seed "$seed" --init_from "$init" $TRAIN_ARGS
      else
        python train.py --block_type "$variant" --seed "$seed" --init_from "$init" $TRAIN_ARGS
      fi
    fi

    if [[ -f evaluate.py ]]; then python evaluate.py --out_dir "$out"; fi
    if [[ -f benchmark.py ]]; then python benchmark.py --out_dir "$out"; fi
  done
done

if [[ -f summarize.py ]]; then python summarize.py; fi
