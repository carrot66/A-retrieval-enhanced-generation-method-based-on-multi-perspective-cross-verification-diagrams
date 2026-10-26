# QPEF-RAG Paper Experiment Report

All values below are from the paper-facing zero-shot release and use the
TruthfulRAG-compatible ACC evaluator.

## Main conflict results

| Model | FaithEval | RealtimeQA | MuSiQue-Negative | SQuAD-Negative | Avg. ACC |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 83.80 | 87.61 | 88.43 | 88.86 | 87.18 |
| Qwen2.5-7B-Instruct | 76.70 | 92.92 | 86.74 | 84.45 | 85.20 |
| Mistral-7B-Instruct-v0.3 | 84.30 | 88.50 | 86.85 | 87.17 | 86.70 |

## Llama component ablation

| Variant | FaithEval | RealtimeQA | MuSiQue-Neg | SQuAD-Neg | Avg. |
|---|---:|---:|---:|---:|---:|
| Full QPEF-RAG | 83.80 | 87.61 | 88.43 | 88.86 | 87.18 |
| w/o dense semantic matching | 82.20 | 87.61 | 87.70 | 88.75 | 86.56 |
| w/o question-conditioned NLI | 81.40 | 87.61 | 87.30 | 88.58 | 86.22 |
| w/o both semantic branches | 80.60 | 87.61 | 86.85 | 87.96 | 85.76 |
| w/o channel capacity | 83.80 | 87.61 | 88.32 | 88.86 | 87.15 |
| w/o conflict-conditioned filtering | 81.90 | 87.61 | 88.32 | 88.81 | 86.66 |
| Evidence verification only | 80.50 | 84.07 | 87.70 | 88.07 | 85.09 |
| Direct graph decision only | 80.70 | 84.07 | 87.70 | 88.07 | 85.14 |
| Standard RAG | 66.60 | 84.07 | 64.45 | 68.12 | 70.81 |
| Same-budget majority | 77.10 | 76.99 | 87.58 | 88.13 | 82.45 |
| Channel-grouped vote | 76.10 | 77.88 | 86.79 | 86.66 | 81.86 |

## RealtimeQA sufficiency

| Model | Without requirement | Full QPEF-RAG | Answerable | Abstention-gold |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 82.30 | 87.61 | 59/63 (93.65) | 40/50 (80.00) |
| Qwen2.5-7B-Instruct | 90.27 | 92.92 | 60/63 (95.24) | 45/50 (90.00) |
| Mistral-7B-Instruct-v0.3 | 64.60 | 88.50 | 59/63 (93.65) | 41/50 (82.00) |

The JSON report retains per-sample aggregate fields and confidence intervals for
the six paper datasets. Published baseline values are not locally rerun data.