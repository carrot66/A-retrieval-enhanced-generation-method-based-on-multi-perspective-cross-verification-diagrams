#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8004/v1}"
MODEL="${MODEL:-llama}"
PROMPT_FORMAT="${PROMPT_FORMAT:-llama}"
WORKERS=12
RUN_ROOT="$ROOT/artifacts/source_ablation"
INPUT_ROOT="$RUN_ROOT/inputs"
SOURCE_ROOT="$RUN_ROOT/source/$MODEL"
SEMANTIC_ROOT="$RUN_ROOT/semantic"
LOG_ROOT="$RUN_ROOT/logs"

export PYTHONPATH="$ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing source-ablation artifacts: $RUN_ROOT" >&2
  echo "Move or remove that directory before starting a fresh run." >&2
  exit 1
fi
mkdir -p "$INPUT_ROOT" "$SOURCE_ROOT" "$SEMANTIC_ROOT" "$LOG_ROOT"

dataset_file() {
  case "$1" in
    faitheval) echo "faitheval_data.json" ;;
    musique_golden) echo "musique_golden.json" ;;
    squad_golden) echo "squad_golden.json" ;;
    timeqa) echo "timeqa_2022_nota.json" ;;
    musique_negative) echo "musique_negative.json" ;;
    squad_negative) echo "squad_negative.json" ;;
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
    --dataset "$dataset" --dataset_path "$inference" --mode plain \
    --output "$plain" --base_url "$BASE_URL" --model "$MODEL" \
    --prompt_format "$PROMPT_FORMAT" --workers "$WORKERS" --seed 20260824 \
    --decision_only >"$LOG_ROOT/$dataset.plain.log" 2>&1

  "$PYTHON" "$ROOT/code/source_generation/source_runner.py" \
    --dataset "$dataset" --dataset_path "$inference" --mode graph \
    --base_input "$plain" --output "$graph" --base_url "$BASE_URL" \
    --model "$MODEL" --prompt_format "$PROMPT_FORMAT" --workers "$WORKERS" \
    --seed 20260824 --decision_only >"$LOG_ROOT/$dataset.graph.log" 2>&1

  "$PYTHON" -m ugcp_v5.semantic_bridge \
    --dataset_path "$inference" --dataset_name "$dataset" \
    --output "$SEMANTIC_ROOT/$dataset.bridge.jsonl" --workers "$WORKERS" \
    --local_device cpu --query_device cuda --query_batch_size 32 \
    >"$LOG_ROOT/$dataset.semantic.log" 2>&1
done

"$PYTHON" -m ugcp_v5.source_channel_ablation \
  --root "$ROOT" --model "$MODEL" --workers "$WORKERS" \
  >"$LOG_ROOT/analysis.log" 2>&1

echo "Source-channel ablation complete: $ROOT/artifacts/results/SOURCE_CHANNEL_ABLATION.md"
