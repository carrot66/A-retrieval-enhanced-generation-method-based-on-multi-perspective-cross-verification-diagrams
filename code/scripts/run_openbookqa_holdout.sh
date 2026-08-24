#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-$ROOT/../../.venv/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8003/v1}"
MODEL="${MODEL:-llama}"
PROMPT_FORMAT="${PROMPT_FORMAT:-llama}"
PHASE="${PHASE:-all}"
LOG_ROOT="$ROOT/artifacts/logs/openbookqa/$MODEL"
SOURCE_ROOT="$ROOT/artifacts/source/$MODEL"
SEMANTIC_ROOT="$ROOT/artifacts/semantic"
PROJECTION_ROOT="$ROOT/artifacts/projections/$MODEL"
PROTOCOL_ROOT="$ROOT/artifacts/holdout_protocol"

mkdir -p "$LOG_ROOT" "$SOURCE_ROOT" "$SEMANTIC_ROOT" "$PROJECTION_ROOT" "$PROTOCOL_ROOT"

prepare_variant() {
  local name="$1"
  local data_file="$2"
  local inference="$PROTOCOL_ROOT/$name.inference.json"
  local answer_key="$PROTOCOL_ROOT/$name.answer_key.json"
  local isolation="$PROTOCOL_ROOT/$name.isolation.json"
  local bridge="$SEMANTIC_ROOT/$name.bridge.jsonl"

  "$PYTHON" "$ROOT/code/data_preparation/isolate_holdout_labels.py" \
    --source "$data_file" --inference_output "$inference" \
    --answer_key_output "$answer_key" --metadata_output "$isolation" \
    2>&1 | tee "$LOG_ROOT/$name.isolation.log"

  PYTHONPATH="$ROOT/code/src" "$PYTHON" -m ugcp_v5.semantic_bridge \
    --dataset_path "$inference" --dataset_name "$name" --output "$bridge" \
    --workers 12 --local_device cpu --query_device cpu \
    2>&1 | tee "$LOG_ROOT/$name.semantic.log"
}

infer_variant() {
  local name="$1"
  local inference="$PROTOCOL_ROOT/$name.inference.json"
  local answer_key="$PROTOCOL_ROOT/$name.answer_key.json"
  local plain="$SOURCE_ROOT/$name.plain.jsonl"
  local graph="$SOURCE_ROOT/$name.graph.jsonl"
  local bridge="$SEMANTIC_ROOT/$name.bridge.jsonl"
  local projection="$PROJECTION_ROOT/$name.jsonl"

  "$PYTHON" "$ROOT/code/source_generation/holdout_runner.py" \
    --dataset_path "$inference" --dataset_name "$name" --mode plain \
    --base_url "$BASE_URL" --model "$MODEL" --prompt_format "$PROMPT_FORMAT" \
    --workers 12 --output "$plain" --no_resume \
    2>&1 | tee "$LOG_ROOT/$name.plain.log"

  "$PYTHON" "$ROOT/code/source_generation/holdout_runner.py" \
    --dataset_path "$inference" --dataset_name "$name" --mode graph \
    --base_input "$plain" --base_url "$BASE_URL" --model "$MODEL" \
    --prompt_format "$PROMPT_FORMAT" --workers 12 --output "$graph" --no_resume \
    2>&1 | tee "$LOG_ROOT/$name.graph.log"

  PYTHONPATH="$ROOT/code/src" "$PYTHON" -m ugcp_v5.replay \
    --dataset "$name" --inference_dataset_path "$inference" \
    --answer_key_path "$answer_key" --model "$MODEL" \
    --plain "$plain" --graph "$graph" --bridge "$bridge" \
    --output "$projection" --workers 12 \
    2>&1 | tee "$LOG_ROOT/$name.replay.log"
}

prepare_all() {
  prepare_variant openbookqa_holdout "$ROOT/data/openbookqa_test.json"
  prepare_variant openbookqa_permuted "$ROOT/data/openbookqa_test_permuted.json"
  prepare_variant openbookqa_distractor "$ROOT/data/openbookqa_test_distractor.json"
}

infer_all() {
  infer_variant openbookqa_holdout
  infer_variant openbookqa_permuted
  infer_variant openbookqa_distractor
}

analyze() {
  PYTHONPATH="$ROOT/code/src" "$PYTHON" -m ugcp_v5.analyze_holdout \
    --base "$PROJECTION_ROOT/openbookqa_holdout.jsonl" \
    --permuted "$PROJECTION_ROOT/openbookqa_permuted.jsonl" \
    --distractor "$PROJECTION_ROOT/openbookqa_distractor.jsonl" \
    --model "$MODEL" \
    --source_dir "$SOURCE_ROOT" \
    --output "$ROOT/artifacts/results/HOLDOUT_RESULTS_$MODEL.json" \
    --bootstrap 10000 --seed 20260710 \
    2>&1 | tee "$LOG_ROOT/analyze.log"
}

case "$PHASE" in
  prepare)
    prepare_all
    ;;
  inference)
    infer_all
    ;;
  analyze)
    analyze
    ;;
  all)
    prepare_all
    infer_all
    analyze
    ;;
  *)
    printf 'Unsupported PHASE: %s\n' "$PHASE" >&2
    exit 2
    ;;
esac
