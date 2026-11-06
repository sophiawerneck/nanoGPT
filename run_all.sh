#!/usr/bin/env bash
# Script principal de execução: dados -> treino de todas as variantes e sementes.
#
# Fluxo de funcionamento:
# - Preparação de dados: Identifica o dataset configurado e aciona o script de 
#   preparação correspondente (ex: prepare_c4.py).
# - Orquestração de treino: Itera sobre todas as sementes (SEEDS) e variantes de 
#   bloco (VARIANTS), executando o pipeline para cada permutação.
# - Verifica os artefatos no diretório de saída (out/): pula execuções já finalizadas (com summary.json) e retoma automaticamente 
#   as interrompidas (se ckpt.pt existir usando --init_from resume).
# - Transita automaticamente entre o uso de uma única GPU 
#   (python) e multi-GPU via DDP (torchrun) com base na variável NPROC.
# - Ao finalizar o treino de uma variante, executa a validação final em conjunto retido (evaluate.py) e a medição de 
#   latência de inferência (benchmark.py).
# - Agregação (Summarize): Após o término de todas as permutações, processa todos os 
#   arquivos gerados para criar as tabelas comparativas e gráficos finais (summarize.py).
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

  python evaluate.py --out_dir "$out"
  python benchmark.py --out_dir "$out"
}

summarize_results() {
  python summarize.py --out_root "out/${DATASET}"
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
