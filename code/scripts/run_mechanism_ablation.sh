#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8004/v1}"
MODEL="${MODEL:-llama}"
PROMPT_FORMAT="${PROMPT_FORMAT:-llama}"
WORKERS=12
RUN_ROOT="$ROOT/artifacts/mechanism_ablation"
INPUT_ROOT="$RUN_ROOT/inputs"
SOURCE_ROOT="$RUN_ROOT/source/$MODEL"
SEMANTIC_ROOT="$RUN_ROOT/semantic"
LOG_ROOT="$RUN_ROOT/logs"

export PYTHONPATH="$ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing mechanism-ablation artifacts: $RUN_ROOT" >&2
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
  esac
}

DATASETS=(faitheval musique_golden squad_golden timeqa musique_negative squad_negative)
VARIANTS=(full without_query_semantic without_local_semantic without_semantic_bridge without_provenance_capacity shared_backbone_capacity without_typed_topology without_directional_cut without_risk_projection without_idk_protection scope_only plain_context majority_same_budget provenance_grouped_vote)

for dataset in "${DATASETS[@]}"; do
  "$PYTHON" "$ROOT/code/data_preparation/isolate_holdout_labels.py" \
    --source "$ROOT/data/$(dataset_file "$dataset")" \
    --inference_output "$INPUT_ROOT/$dataset.inference.json" \
    --answer_key_output "$INPUT_ROOT/$dataset.answer_key.json" \
    --metadata_output "$INPUT_ROOT/$dataset.isolation.json"
done

for variant in "${VARIANTS[@]}"; do
  seed=20260824
  mkdir -p "$SOURCE_ROOT/$variant" "$SEMANTIC_ROOT/$variant"
  for dataset in "${DATASETS[@]}"; do
    inference="$INPUT_ROOT/$dataset.inference.json"
    plain="$SOURCE_ROOT/$variant/$dataset.plain.jsonl"
    graph="$SOURCE_ROOT/$variant/$dataset.graph.jsonl"
    "$PYTHON" "$ROOT/code/source_generation/source_runner.py" \
      --dataset "$dataset" --dataset_path "$inference" --mode plain \
      --output "$plain" --base_url "$BASE_URL" --model "$MODEL" \
      --prompt_format "$PROMPT_FORMAT" --workers "$WORKERS" --seed "$seed" \
      --decision_only --no_resume >"$LOG_ROOT/$variant.$dataset.plain.log" 2>&1
    "$PYTHON" "$ROOT/code/source_generation/source_runner.py" \
      --dataset "$dataset" --dataset_path "$inference" --mode graph \
      --base_input "$plain" --output "$graph" --base_url "$BASE_URL" \
      --model "$MODEL" --prompt_format "$PROMPT_FORMAT" --workers "$WORKERS" \
      --seed "$seed" --decision_only --no_resume \
      >"$LOG_ROOT/$variant.$dataset.graph.log" 2>&1
    "$PYTHON" -m ugcp_v5.semantic_bridge \
      --dataset_path "$inference" --dataset_name "$dataset" \
      --output "$SEMANTIC_ROOT/$variant/$dataset.bridge.jsonl" --workers "$WORKERS" \
      --local_device cpu --query_device cuda --query_batch_size 32 \
      >"$LOG_ROOT/$variant.$dataset.semantic.log" 2>&1
  done
done

"$PYTHON" -m ugcp_v5.mechanism_ablation \
  --root "$ROOT" --run_root "$RUN_ROOT" --model "$MODEL" --workers "$WORKERS" \
  >"$LOG_ROOT/analysis.log" 2>&1

echo "Mechanism ablation complete: $ROOT/artifacts/results/MECHANISM_ABLATION.json"
