# QPEF-RAG: Query-aware Provenance-guided Evidence Fusion

This repository contains the paper-facing, training-free implementation and
experiment data for QPEF-RAG. The internal source package retains the
ugcp_v5 and QPCF identifiers for compatibility; the paper-facing method name
is QPEF-RAG.

## Experiment scope

The release covers six evaluation files:

- FaithEval
- RealtimeQA (data/timeqa_2022_nota.json)
- MuSiQue-Golden and MuSiQue-Negative
- SQuAD-Golden and SQuAD-Negative

Llama runs all six datasets. Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3 run the four conflict datasets. The reported runs are
zero-shot, use temperature 0, and use workers=12.

## Paper-facing results

| Model | FaithEval | RealtimeQA | MuSiQue-Neg | SQuAD-Neg | Avg. conflict ACC |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 83.80 | 87.61 | 88.43 | 88.86 | 87.18 |
| Qwen2.5-7B-Instruct | 76.70 | 92.92 | 86.74 | 84.45 | 85.20 |
| Mistral-7B-Instruct-v0.3 | 84.30 | 88.50 | 86.85 | 87.17 | 86.70 |

The complete report is in artifacts/results/PAPER_REPORT.json and
artifacts/results/PAPER_REPORT.md. Published baseline point estimates are
cited in the manuscript and are not stored as local rerun traces.

## Layout

- code/src/ugcp_v5/: graph projection, semantic grounding, and evaluation.
- code/source_generation/: source-generation and evaluator code.
- code/data_preparation/: input preparation and label separation.
- code/scripts/replay_formal.sh: replay retained source traces into per-sample decisions.
- data/: the six evaluation files used by the paper.
- artifacts/source/: retained model source traces.
- artifacts/semantic/: retained frozen semantic-grounding traces.
- artifacts/projections/: retained per-sample decisions.
- artifacts/results/: paper-facing aggregate results.

## Reproduction

Install code/requirements.txt, provide an OpenAI-compatible inference endpoint
when regenerating source traces, and run:

    bash code/scripts/replay_formal.sh

The replay script uses the retained source and semantic traces and does not
make new model calls unless those traces are regenerated. Gold labels are used
only after the final option ID has been fixed.