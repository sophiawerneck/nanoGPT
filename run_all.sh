#!/usr/bin/env bash
# Script principal de execução: dados -> treino de todas as variantes e sementes.
# execuções concluídas são puladas e as interrompidas são retomadas.
#
# As etapas de avaliação (Membro 4), benchmark e agregação (Membro 3) rodam
# automaticamente quando os respectivos scripts existirem no repositório:
#   evaluate.py  --out_dir <execução>   -> perda/perplexidade final
#   benchmark.py --out_dir <execução>   -> latência de inferência
#   summarize.py --out_root <pasta>     -> tabelas e speedup
#
# Variáveis de ambiente (opcionais):
#   DATASET=c4               c4 (experimento) ou shakespeare_char (teste rápido)
#   SEEDS="1337 1338 1339"   sementes (padrão: 3)
#   VARIANTS="seq inv par"   variantes
#   NPROC=1                  GPUs por execução (usa torchrun se > 1)
#   TRAIN_ARGS=""            argumentos extras para train.py (ex.: --peak_tflops 989)
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${DATASET:-c4}"
SEEDS="${SEEDS:-1337 1338 1339}"
VARIANTS="${VARIANTS:-seq inv par}"
NPROC="${NPROC:-1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

prepare_data() {
  case "$DATASET" in
    c4)
      python prepare_c4.py ;;
    shakespeare_char)
      [[ -f data/shakespeare_char/train.bin ]] || python data/shakespeare_char/prepare.py ;;
    *)
      echo "DATASET inválido: $DATASET (use c4 ou shakespeare_char)" >&2
      exit 2 ;;
  esac
}

train_and_evaluate() {
  local seed="$1"
  local variant="$2"
  local out="out/${DATASET}/${variant}_s${seed}"

  if [[ -f "$out/summary.json" ]]; then
    echo ">> $out já treinado."
  else
    local init="scratch"
    [[ -f "$out/ckpt.pt" ]] && init="resume"
    echo ">> Treinando $variant (dataset $DATASET, semente $seed, $init)"
    
    local args=(--dataset "$DATASET" --block_type "$variant" --seed "$seed" --init_from "$init")
    if [[ "$NPROC" -gt 1 ]]; then
      torchrun --standalone --nproc_per_node="$NPROC" train.py "${args[@]}" $TRAIN_ARGS
    else
      python train.py "${args[@]}" $TRAIN_ARGS
    fi
  fi

  if [[ -f evaluate.py ]]; then python evaluate.py --out_dir "$out"; fi
  if [[ -f benchmark.py ]]; then python benchmark.py --out_dir "$out"; fi
}

summarize_results() {
  if [[ -f summarize.py ]]; then 
    python summarize.py --out_root "out/${DATASET}"
  fi
}

main() {
  prepare_data

  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      train_and_evaluate "$seed" "$variant"
    done
  done

  summarize_results
}

main "$@"
