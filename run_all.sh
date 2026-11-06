
#!/usr/bin/env bash
# Script principal de execução: dados -> treino de todas as variantes e sementes.
#
# Fluxo de funcionamento:
# - Preparação de dados: identifica o dataset configurado e aciona o script
#   de preparação correspondente (ex: prepare_c4.py).
# - Orquestração de treino: itera sobre todas as sementes (SEEDS) e variantes
#   de bloco (VARIANTS), executando o pipeline para cada combinação.
# - Verifica os artefatos no diretório de saída (out/): pula execuções já
#   finalizadas (com summary.json) e retoma as interrompidas (com ckpt.pt).
# - Transita automaticamente entre uma GPU (python) e múltiplas GPUs via DDP
#   (torchrun), com base na variável NPROC.
# - Após o treinamento, executa o benchmark de latência de inferência
#   (bench.py).
# - Agregação (Summarize): processa os arquivos gerados para criar as tabelas
#   comparativas e os gráficos finais (summarize.py).
#
# Variáveis de ambiente (opcionais):
#   DATASET=c4               c4 (experimento) ou shakespeare_char (teste)
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
      if [[ ! -f data/c4/train.bin ||
            ( ! -f data/c4/validation.bin &&
              ! -f data/c4/val.bin ) ]]; then
        echo "Preparando o C4..."
        python prepare_c4.py
      else
        echo "Dados do C4 já presentes. Pulando preparação."
      fi
      ;;

    shakespeare_char)
      if [[ ! -f data/shakespeare_char/train.bin ||
            ! -f data/shakespeare_char/meta.pkl ||
            ( ! -f data/shakespeare_char/val.bin &&
              ! -f data/shakespeare_char/validation.bin ) ]]; then
        echo "Preparando Shakespeare..."
        python data/shakespeare_char/prepare.py
      else
        echo "Dados de Shakespeare já presentes. Pulando preparação."
      fi
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
  local out="out/${DATASET}/${variant}_s${seed}"

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

    if [[ "$NPROC" -gt 1 ]]; then
      torchrun --standalone --nproc_per_node="$NPROC" \
        train.py "${args[@]}" $TRAIN_ARGS
    else
      python train.py "${args[@]}" $TRAIN_ARGS
    fi
  fi

  python bench.py --out_dir "$out"
}

summarize_results() {
  python summarize.py --out_root "out/${DATASET}"
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
}

main "$@"
