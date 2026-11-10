#!/usr/bin/env bash
# Script principal de execução: dados -> treino de todas as variantes e sementes.
#
# Fluxo de funcionamento:
# - Preparação de dados: identifica o dataset configurado e aciona o script
#   de preparação correspondente.
# - Orquestração de treino: itera sobre todas as sementes (SEEDS) e variantes
#   de bloco (VARIANTS), executando o pipeline para cada combinação.
# - Verifica os artefatos no diretório de saída: pula execuções já finalizadas
#   (com summary.json) e retoma as interrompidas (com ckpt.pt).
# - Transita automaticamente entre uma GPU (python) e múltiplas GPUs via DDP
#   (torchrun), com base na variável NPROC.
# - Após o treinamento, executa o benchmark de latência de inferência
#   (bench.py).
# - Agregação: processa os arquivos gerados para criar as tabelas
#   comparativas e os gráficos finais (summarize.py).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET="${DATASET:-c4}"
SEEDS="${SEEDS:-1337 1338 1339}"
VARIANTS="${VARIANTS:-seq inv par par_shared}"
NPROC="${NPROC:-1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

OUTPUT_DIR="$(python -c 'from common import OUTPUT_DIR; print(OUTPUT_DIR)')"
INPUT_DIR="$(python -c 'from common import INPUT_DIR; print(INPUT_DIR)')"

DATASET_OUT="${OUTPUT_DIR}/${DATASET}"
C4_INPUT="${INPUT_DIR}/c4"

mkdir -p "$DATASET_OUT"
exec > >(tee -a "${DATASET_OUT}/run_all.log") 2>&1

echo "Diretório de entrada: ${INPUT_DIR}"
echo "Diretório de saída: ${DATASET_OUT}"
echo "Dataset: ${DATASET}"
echo "Sementes: ${SEEDS}"
echo "Variantes: ${VARIANTS}"
echo "NPROC: ${NPROC}"

EXTRA_TRAIN_ARGS=()
if [[ -n "$TRAIN_ARGS" ]]; then
  read -r -a EXTRA_TRAIN_ARGS <<< "$TRAIN_ARGS"
fi

prepare_data() {
  case "$DATASET" in
    c4)
      if [[ ! -f "$C4_INPUT/train.bin" ||
            ( ! -f "$C4_INPUT/validation.bin" &&
              ! -f "$C4_INPUT/val.bin" ) ]]; then
        echo "Preparando o C4..."
        python data/c4/prepare.py
      else
        echo "Dados do C4 já presentes. Pulando preparação."
      fi
      ;;

    shakespeare_char)
      echo "A preparação de shakespeare_char ainda precisa ser adaptada para INPUT_DIR." >&2
      exit 2
      ;;

    *)
      echo "DATASET inválido: $DATASET (use c4 ou shakespeare_char)" >&2
      exit 2
      ;;
  esac
}

train_and_evaluate() {
  local seed="$1"
  local variant="$2"
  local out="${DATASET_OUT}/${variant}_s${seed}"

  if [[ -f "$out/summary.json" ]]; then
    echo ">> $out já treinado."
  else
    local init="scratch"
    [[ -f "$out/ckpt.pt" ]] && init="resume"

    echo ">> Treinando $variant (dataset $DATASET, semente $seed, $init)"

    local args=(
      --dataset "$DATASET"
      --block_type "$variant"
      --seed "$seed"
      --init_from "$init"
      --out_dir "$out"
    )

    args+=("${EXTRA_TRAIN_ARGS[@]}")

    if [[ "$NPROC" -gt 1 ]]; then
      torchrun --standalone --nproc_per_node="$NPROC" \
        train.py "${args[@]}"
    else
      python train.py "${args[@]}"
    fi
  fi

  python bench.py --out_dir "$out"
}

summarize_results() {
  python summarize.py --out_root "$DATASET_OUT"
}

main() {
  if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC deve ser um inteiro positivo." >&2
    exit 2
  fi

  prepare_data

  for seed in $SEEDS; do
    for variant in $VARIANTS; do
      train_and_evaluate "$seed" "$variant"
    done
  done

  summarize_results

  echo "Pipeline concluído."
  echo "Resultados salvos em: $DATASET_OUT"
}

main "$@"
