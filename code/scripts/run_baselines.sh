#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="$ROOT/code"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-$ROOT}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
WORKERS=12
DATASETS=(faitheval timeqa musique_negative squad_negative)
METHODS=(without_rag rag kre)
read -r -a MODELS <<< "${MODELS:-llama qwen mistral}"

export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

dataset_file() {
  case "$1" in
    faitheval) echo "faitheval_data.json" ;;
    timeqa) echo "timeqa_2022_nota.json" ;;
    musique_negative) echo "musique_negative.json" ;;
    squad_negative) echo "squad_negative.json" ;;
    *) echo "unsupported dataset: $1" >&2; return 1 ;;
  esac
}

base_url() {
  case "$1" in
    llama) echo "${LLAMA_BASE_URL:-http://127.0.0.1:8004/v1}" ;;
    qwen) echo "${QWEN_BASE_URL:-http://127.0.0.1:8003/v1}" ;;
    mistral) echo "${MISTRAL_BASE_URL:-http://127.0.0.1:8002/v1}" ;;
    *) echo "unsupported model: $1" >&2; return 1 ;;
  esac
}

served_model() {
  case "$1" in
    llama) echo "${LLAMA_SERVED_MODEL:-llama}" ;;
    qwen) echo "${QWEN_SERVED_MODEL:-qwen}" ;;
    mistral) echo "${MISTRAL_SERVED_MODEL:-Mistral}" ;;
    *) echo "unsupported model: $1" >&2; return 1 ;;
  esac
}

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      output="$RUN_ROOT/artifacts/baselines/$model/$dataset/$method.jsonl"
      mkdir -p "$(dirname "$output")"
      "$PYTHON" -m ugcp_v5.baseline_runner \
        --method "$method" \
        --dataset "$dataset" \
        --dataset_path "$DATA_ROOT/$(dataset_file "$dataset")" \
        --model "$(served_model "$model")" \
        --base_url "$(base_url "$model")" \
        --output "$output" \
        --workers "$WORKERS"
    done
  done
done

echo "Baseline source traces complete: $RUN_ROOT/artifacts/baselines"
