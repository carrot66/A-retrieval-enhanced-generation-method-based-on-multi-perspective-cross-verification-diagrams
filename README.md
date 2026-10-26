# QPEF-RAG: Query-aware Provenance-guided Evidence Fusion

This repository contains the paper-facing, training-free implementation and
experiment release for QPEF-RAG. The internal code package retains the
ugcp_v5 and QPCF identifiers for compatibility with the released scripts; the
paper-facing method name is QPEF-RAG.

## Experiment scope

The release covers six fixed evaluation files:

- FaithEval
- RealtimeQA (data/timeqa_2022_nota.json)
- MuSiQue-Golden and MuSiQue-Negative
- SQuAD-Golden and SQuAD-Negative

Llama runs all six datasets. Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3 run the four conflict datasets. All reported runs are
zero-shot, use temperature 0, and use workers=12. Gold labels are joined only
after option selection.

## Paper-facing results

| Model | FaithEval | RealtimeQA | MuSiQue-Neg | SQuAD-Neg | Avg. conflict ACC |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 83.80 | 87.61 | 88.43 | 88.86 | 87.18 |
| Qwen2.5-7B-Instruct | 76.70 | 92.92 | 86.74 | 84.45 | 85.20 |
| Mistral-7B-Instruct-v0.3 | 84.30 | 88.50 | 86.85 | 87.17 | 86.70 |

The complete paper-facing result report is in
artifacts/results/PAPER_REPORT.json and artifacts/results/PAPER_REPORT.md.
Published baseline point estimates are cited in the manuscript and are not
treated as locally reproduced per-sample data.

## Layout

- code/src/ugcp_v5/: graph projection, semantic grounding, evaluation, and analysis.
- code/source_generation/: source-generation and evaluator code.
- code/reporting/: paper report, audit, and release-manifest builders.
- code/scripts/replay_formal.sh: replays retained source traces and rebuilds paper-facing projections.
- data/: the six evaluation files used by the paper.
- artifacts/source/: retained LLM source traces for the six datasets.
- artifacts/semantic/: retained frozen semantic-grounding traces.
- artifacts/projections/: retained per-sample decisions for the 14 model-dataset pairs.
- artifacts/results/: paper-facing aggregate results and the IDK sufficiency analysis.

## Reproduction

Install code/requirements.txt, provide an OpenAI-compatible inference endpoint
for source generation, and run:

    bash code/scripts/replay_formal.sh

When only retained traces are used, the script rebuilds decisions without
making new model calls. The evaluator follows the official TruthfulRAG ACC
normalization used by the manuscript.

## Data boundary

OpenBookQA, holdout stress tests, development probes, post-hoc audits, and
fresh-release variants outside the manuscript are intentionally excluded from
this paper-facing release. They remain available in Git history if needed for
separate analysis.