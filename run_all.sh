#!/usr/bin/env bash
# Script principal de execução: dados -> treino de todas as variantes e sementes.
#
# Fluxo de funcionamento:
# - Preparação de dados: identifica o dataset configurado e aciona o script
#   de preparação correspondente. A preparação só é pulada quando os
#   *.state.json indicam que ela terminou ("complete": true).
# - Orquestração de treino: itera sobre todas as sementes (SEEDS) e variantes
#   de bloco (VARIANTS), executando o pipeline para cada combinação.
# - Verifica os artefatos no diretório de saída: pula execuções já finalizadas
#   (com summary.json) e retoma as interrompidas (com ckpt.pt).
# - Transita automaticamente entre uma GPU (python) e múltiplas GPUs via DDP
#   (torchrun), com base na variável NPROC.
# - Após o treinamento, executa o benchmark de latência de inferência
#   (bench.py).
# - Réplica opcional: REPLICA="<variante> <semente>" repete uma execução já
#   feita, em diretório separado, para medir o ruído de não determinismo.
# - Agregação: processa os arquivos gerados para criar as tabelas
#   comparativas e os gráficos finais (summarize.py).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET="${DATASET:-c4}"
SEEDS="${SEEDS:-1337 1338 1339}"
VARIANTS="${VARIANTS:-seq inv par par_shared}"
NPROC="${NPROC:-1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
REPLICA="${REPLICA:-}"

OUTPUT_DIR="$(python -c 'from common import OUTPUT_DIR; print(OUTPUT_DIR)')"
INPUT_DIR="$(python -c 'from common import INPUT_DIR; print(INPUT_DIR)')"

DATASET_OUT="${OUTPUT_DIR}/${DATASET}"
# Fora de DATASET_OUT para que summarize.py não conte a réplica como semente extra.
REPLICA_OUT="${OUTPUT_DIR}/${DATASET}_replica"
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

# Estado da preparação do C4, lido dos *.state.json gravados pelo prepare.py:
#   0 = train e validation completos ("complete": true e .bin presente);
#   1 = ausente ou incompleto (o prepare.py retoma de onde parou);
#   2 = .bin presente sem *.state.json (não há como confirmar que terminou).
c4_data_status() {
  python - "$C4_INPUT" <<'PY'
import json, os, sys

data_dir = sys.argv[1]
status = 0
for split in ("train", "validation"):
    bin_path = os.path.join(data_dir, f"{split}.bin")
    state_path = os.path.join(data_dir, f"{split}.state.json")
    if not os.path.exists(state_path):
        status = max(status, 2 if os.path.exists(bin_path) else 1)
        continue
    with open(state_path, encoding="utf-8") as f:
        complete = json.load(f).get("complete")
    if not (complete and os.path.exists(bin_path)):
        status = max(status, 1)
sys.exit(status)
PY
}

prepare_data() {
  case "$DATASET" in
    c4)
      local status=0
      c4_data_status || status=$?
      case "$status" in
        0)
          echo "Dados do C4 completos (*.state.json com complete=true). Pulando preparação."
          ;;
        1)
          echo "Preparando o C4 (ausente ou incompleto)..."
          python data/c4/prepare.py
          ;;
        *)
          echo "Há .bin em $C4_INPUT sem o *.state.json correspondente: não é possível" >&2
          echo "confirmar que a preparação terminou. Restaure os *.state.json e *_meta.json" >&2
          echo "gerados pelo prepare.py ou prepare os dados novamente." >&2
          exit 2
          ;;
      esac
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
  local root="${3:-$DATASET_OUT}"
  local out="${root}/${variant}_s${seed}"

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
  
  # Gera 5 amostras de texto e salva num arquivo txt
  echo ">> Gerando amostras de texto para $variant..."
  python sample.py --out_dir "$out" --num_samples 5 > "$out/samples_text.txt"
}


# Repete uma execução com a mesma semente e compara a loss final do holdout
# com a da execução original: a diferença estima o ruído de não determinismo.
run_replica() {
  local variant seed extra
  read -r variant seed extra <<< "$REPLICA"
  if [[ -z "${variant:-}" || -z "${seed:-}" || -n "${extra:-}" ]]; then
    echo "REPLICA deve ter a forma \"<variante> <semente>\" (ex.: REPLICA=\"seq 1337\")." >&2
    exit 2
  fi
  local original="${DATASET_OUT}/${variant}_s${seed}/summary.json"
  if [[ ! -f "$original" ]]; then
    echo "REPLICA: execução original não encontrada ($original)." >&2
    exit 2
  fi

  train_and_evaluate "$seed" "$variant" "$REPLICA_OUT"

  python - "$original" "${REPLICA_OUT}/${variant}_s${seed}/summary.json" <<'PY'
import json, math, sys

def holdout_loss(path):
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    loss = summary.get("val_holdout_loss_final")
    return loss if loss is not None else math.log(summary["val_holdout_ppl_final"])

original, replica = holdout_loss(sys.argv[1]), holdout_loss(sys.argv[2])
print(f"Réplica (mesma semente): loss do holdout original {original:.4f} | réplica {replica:.4f} "
      f"| diferença {replica - original:+.4f} nats (ruído de não determinismo)")
PY
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

  if [[ -n "$REPLICA" ]]; then
    run_replica
  fi

  summarize_results

  echo "Pipeline concluído."
  echo "Resultados salvos em: $DATASET_OUT"
}

main "$@"
