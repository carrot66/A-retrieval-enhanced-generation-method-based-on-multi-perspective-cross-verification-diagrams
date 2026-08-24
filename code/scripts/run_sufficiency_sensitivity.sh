#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8004/v1}"
MODEL="${MODEL:-llama}"
PROMPT_FORMAT="${PROMPT_FORMAT:-llama}"
WORKERS=12
SENSITIVITY_ROOT="$ROOT/artifacts/sensitivity"
INPUT_ROOT="$SENSITIVITY_ROOT/inputs"
SOURCE_ROOT="$SENSITIVITY_ROOT/source/$MODEL"
SEMANTIC_ROOT="$SENSITIVITY_ROOT/semantic"
LOG_ROOT="$SENSITIVITY_ROOT/logs"

export PYTHONPATH="$ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$INPUT_ROOT" "$SOURCE_ROOT" "$SEMANTIC_ROOT" "$LOG_ROOT"

dataset_file() {
  case "$1" in
    faitheval) echo "faitheval_data.json" ;;
    timeqa) echo "timeqa_2022_nota.json" ;;
    musique_negative) echo "musique_negative.json" ;;
    squad_negative) echo "squad_negative.json" ;;
    musique_golden) echo "musique_golden.json" ;;
    squad_golden) echo "squad_golden.json" ;;
    *) echo "unsupported dataset: $1" >&2; return 1 ;;
  esac
}

for dataset in faitheval musique_golden squad_golden timeqa musique_negative squad_negative; do
  source_file="$ROOT/data/$(dataset_file "$dataset")"
  inference="$INPUT_ROOT/$dataset.inference.json"
  answer_key="$INPUT_ROOT/$dataset.answer_key.json"
  isolation="$INPUT_ROOT/$dataset.isolation.json"
  "$PYTHON" "$ROOT/code/data_preparation/isolate_holdout_labels.py" \
    --source "$source_file" \
    --inference_output "$inference" \
    --answer_key_output "$answer_key" \
    --metadata_output "$isolation"

  plain="$SOURCE_ROOT/$dataset.plain.jsonl"
  graph="$SOURCE_ROOT/$dataset.graph.jsonl"
  "$PYTHON" "$ROOT/code/source_generation/source_runner.py" \
    --dataset "$dataset" \
    --dataset_path "$inference" \
    --mode plain \
    --output "$plain" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --prompt_format "$PROMPT_FORMAT" \
    --workers "$WORKERS" \
    --seed 20260817 \
    --decision_only \
    >"$LOG_ROOT/$dataset.plain.log" 2>&1

  "$PYTHON" "$ROOT/code/source_generation/source_runner.py" \
    --dataset "$dataset" \
    --dataset_path "$inference" \
    --mode graph \
    --base_input "$plain" \
    --output "$graph" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --prompt_format "$PROMPT_FORMAT" \
    --workers "$WORKERS" \
    --seed 20260817 \
    --decision_only \
    >"$LOG_ROOT/$dataset.graph.log" 2>&1

  "$PYTHON" -m ugcp_v5.semantic_bridge \
    --dataset_path "$inference" \
    --dataset_name "$dataset" \
    --output "$SEMANTIC_ROOT/$dataset.bridge.jsonl" \
    --workers "$WORKERS" \
    --local_device cpu \
    --query_device cuda \
    >"$LOG_ROOT/$dataset.semantic.log" 2>&1
done

"$PYTHON" -m ugcp_v5.sufficiency_sensitivity \
  --root "$ROOT" \
  --model "$MODEL" \
  --workers "$WORKERS" \
  --thresholds 2 3 4 \
  >"$LOG_ROOT/analysis.log" 2>&1
