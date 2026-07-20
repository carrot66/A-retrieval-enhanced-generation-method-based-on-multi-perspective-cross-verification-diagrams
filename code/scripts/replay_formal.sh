#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-$(cd "$ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$RUN_ROOT/data}"
BRIDGE_ROOT="${BRIDGE_ROOT:-$RUN_ROOT/artifacts/semantic}"
LLAMA_SOURCE_ROOT="${LLAMA_SOURCE_ROOT:-$RUN_ROOT/artifacts/source/llama}"
LLAMA_GOLDEN_SOURCE_ROOT="${LLAMA_GOLDEN_SOURCE_ROOT:-$LLAMA_SOURCE_ROOT}"
QWEN_SOURCE_ROOT="${QWEN_SOURCE_ROOT:-$RUN_ROOT/artifacts/source/qwen}"
MISTRAL_SOURCE_ROOT="${MISTRAL_SOURCE_ROOT:-$RUN_ROOT/artifacts/source/mistral}"
MISTRAL_ISOLATION_ROOT="$(mktemp -d)"
trap 'rm -rf "$MISTRAL_ISOLATION_ROOT"' EXIT

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

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

run_pair() {
  local model="$1"
  local dataset="$2"
  local source_root="$3"
  local output="$RUN_ROOT/artifacts/projections/$model/$dataset.jsonl"
  mkdir -p "$(dirname "$output")"
  "$PYTHON" -m ugcp_v5.replay \
    --dataset "$dataset" \
    --dataset_path "$DATA_ROOT/$(dataset_file "$dataset")" \
    --model "$model" \
    --plain "$source_root/$dataset.plain.jsonl" \
    --graph "$source_root/$dataset.graph.jsonl" \
    --bridge "$BRIDGE_ROOT/$dataset.bridge.jsonl" \
    --output "$output" \
    --workers 12
}

run_pair_isolated() {
  local model="$1"
  local dataset="$2"
  local source_root="$3"
  local dataset_path="$DATA_ROOT/$(dataset_file "$dataset")"
  local inference="$MISTRAL_ISOLATION_ROOT/$dataset.inference.json"
  local answer_key="$MISTRAL_ISOLATION_ROOT/$dataset.answer_key.json"
  local isolation="$MISTRAL_ISOLATION_ROOT/$dataset.isolation.json"
  local output="$RUN_ROOT/artifacts/projections/$model/$dataset.jsonl"
  mkdir -p "$(dirname "$output")"
  "$PYTHON" "$ROOT/data_preparation/isolate_holdout_labels.py" \
    --source "$dataset_path" \
    --inference_output "$inference" \
    --answer_key_output "$answer_key" \
    --metadata_output "$isolation" >/dev/null
  "$PYTHON" -m ugcp_v5.replay \
    --dataset "$dataset" \
    --inference_dataset_path "$inference" \
    --answer_key_path "$answer_key" \
    --model "$model" \
    --plain "$source_root/$dataset.plain.jsonl" \
    --graph "$source_root/$dataset.graph.jsonl" \
    --bridge "$BRIDGE_ROOT/$dataset.bridge.jsonl" \
    --output "$output" \
    --workers 12
}

for dataset in faitheval musique_golden squad_golden timeqa musique_negative squad_negative; do
  if [[ "$dataset" == *golden ]]; then
    run_pair llama "$dataset" "$LLAMA_GOLDEN_SOURCE_ROOT"
  else
    run_pair llama "$dataset" "$LLAMA_SOURCE_ROOT"
  fi
done
"$PYTHON" -m ugcp_v5.gate \
  --config "$ROOT/configs/experiment.json" --root "$RUN_ROOT" --model llama

for dataset in faitheval timeqa musique_negative squad_negative; do
  run_pair qwen "$dataset" "$QWEN_SOURCE_ROOT"
done
"$PYTHON" -m ugcp_v5.gate \
  --config "$ROOT/configs/experiment.json" --root "$RUN_ROOT" --model qwen

for dataset in faitheval timeqa musique_negative squad_negative; do
  run_pair_isolated mistral "$dataset" "$MISTRAL_SOURCE_ROOT"
done
"$PYTHON" -m ugcp_v5.gate \
  --config "$ROOT/configs/experiment.json" --root "$RUN_ROOT" --model mistral

"$PYTHON" -m ugcp_v5.analyze \
  --config "$ROOT/configs/experiment.json" \
  --root "$RUN_ROOT" \
  --bootstrap 10000 \
  --seed 20260710

"$PYTHON" -m ugcp_v5.paired_counterfactual \
  --config "$ROOT/configs/experiment.json" \
  --root "$RUN_ROOT" \
  --bootstrap 10000 \
  --seed 20260710

"$PYTHON" -m ugcp_v5.provenance_robustness \
  --config "$ROOT/configs/experiment.json" \
  --root "$RUN_ROOT" \
  --replicas 8

"$PYTHON" "$ROOT/reporting/build_paper_report.py" \
  --root "$RUN_ROOT" \
  --output "$RUN_ROOT/artifacts/results/PAPER_REPORT.json"

"$PYTHON" "$ROOT/reporting/audit_release.py" \
  --root "$RUN_ROOT" \
  --output "$RUN_ROOT/artifacts/results/RELEASE_AUDIT.json"

"$PYTHON" "$ROOT/reporting/build_release_manifest.py" \
  --root "$RUN_ROOT"
