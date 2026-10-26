# Reproducibility

## Scope

The paper-facing release evaluates FaithEval, RealtimeQA, MuSiQue-Golden,
MuSiQue-Negative, SQuAD-Golden, and SQuAD-Negative. Llama runs all six files;
Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 run the four conflict files.
The reported experiments are zero-shot with temperature 0 and workers=12.

## Label isolation

code/data_preparation/isolate_holdout_labels.py creates an inference file
without answer fields and a separate answer-key file. The source runner accepts
only the inference file during decision making. Gold answers are joined after
the final option ID has been fixed.

## Replay

code/scripts/replay_formal.sh reads the retained plain/graph source traces and
semantic traces, rebuilds per-sample projections, and writes the aggregate
paper report. It does not use gold labels for routing or decision making.

The main report is written to artifacts/results/PAPER_REPORT.json and
artifacts/results/PAPER_REPORT.md. Dataset-level accuracy follows the official
TruthfulRAG evaluator in code/source_generation/evaluate.py; the manuscript
reports the unweighted mean over the four conflict datasets.

## Environment

- Python 3.10 or newer
- Dependencies in code/requirements.txt
- An OpenAI-compatible endpoint only when regenerating source traces
- Frozen semantic models all-MiniLM-L6-v2 and nli-deberta-v3-small