# UGCP-v5.1-QPCF Unified Paper Report

QPCF-v5.1 is frozen as a training-free, zero-shot conflict-resolution method. Published-paper rows are descriptive point-estimate gates; paired significance is claimed only for included same-sample internal comparisons.

## Conflict and Golden Results

| Model | Dataset | N | ACC | 95% CI | Published point estimate | Margin |
|---|---|---:|---:|---:|---:|---:|
| llama | FaithEval | 1000 | 83.80 | [81.39, 85.95] | 79.80 | +4.00 |
| llama | MuSiQue-Golden | 1772 | 92.72 | [91.42, 93.84] | 85.30 | +7.42 |
| llama | SQuAD-Golden | 1769 | 98.42 | [97.72, 98.90] | 96.60 | +1.82 |
| llama | RealtimeQA | 113 | 87.61 | [80.27, 92.47] | 81.40 | +6.21 |
| llama | MuSiQue-Negative | 1772 | 88.43 | [86.86, 89.84] | 79.90 | +8.53 |
| llama | SQuAD-Negative | 1769 | 88.86 | [87.31, 90.25] | 86.30 | +2.56 |
| qwen | FaithEval | 1000 | 76.70 | [73.98, 79.21] | 73.20 | +3.50 |
| qwen | RealtimeQA | 113 | 92.92 | [86.65, 96.37] | 82.30 | +10.62 |
| qwen | MuSiQue-Negative | 1772 | 86.74 | [85.08, 88.24] | 79.10 | +7.64 |
| qwen | SQuAD-Negative | 1769 | 84.45 | [82.69, 86.07] | 78.70 | +5.75 |
| mistral | FaithEval | 1000 | 84.30 | [81.91, 86.42] | 81.90 | +2.40 |
| mistral | RealtimeQA | 113 | 88.50 | [81.31, 93.15] | 81.40 | +7.10 |
| mistral | MuSiQue-Negative | 1772 | 86.85 | [85.20, 88.35] | 79.30 | +7.55 |
| mistral | SQuAD-Negative | 1769 | 87.17 | [85.53, 88.65] | 85.70 | +1.47 |

## Frozen OpenBookQA Stress Test

| Variant | N | QPCF | Plain RAG | Majority | QPCF - plain | Rescued | Harmed |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 500 | 83.40 | 85.40 | 84.00 | -2.00 | 10 | 20 |
| permuted | 500 | 83.20 | 86.60 | 81.80 | -3.40 | 10 | 27 |
| distractor | 500 | 84.00 | 87.00 | 83.60 | -3.00 | 5 | 20 |

The source audit covers 1500 sample variants and 3720 real model requests with zero missing request, response, or source-label fields.

## Paired Golden-Counterfactual Results

| Pair | Full joint | Plain joint | Majority joint | Grouped-vote joint |
|---|---:|---:|---:|---:|
| MuSiQue | 83.07 | 70.94 | 80.81 | 79.97 |
| SQuAD | 87.56 | 73.88 | 86.55 | 85.02 |

## Cross-Configuration Ablations

| Comparator | Weighted full-minus-comparator | Positive / tied / negative | Holm-significant +/- |
|---|---:|---:|---:|
| without_query_semantic | +1.320 | 10 / 4 / 0 | 4 / 0 |
| without_local_semantic | +1.251 | 13 / 1 / 0 | 4 / 0 |
| without_semantic_bridge | +2.942 | 13 / 1 / 0 | 8 / 0 |
| without_provenance_capacity | +0.069 | 6 / 7 / 1 | 0 / 0 |
| shared_backbone_capacity | -0.440 | 4 / 3 / 7 | 0 / 3 |
| without_typed_topology | +0.183 | 7 / 4 / 3 | 0 / 0 |
| without_directional_cut | +0.171 | 7 / 4 / 3 | 1 / 0 |
| without_risk_projection | +0.863 | 12 / 0 / 2 | 2 / 0 |
| without_idk_protection | +0.194 | 3 / 11 / 0 | 1 / 0 |
| scope_only | +3.205 | 12 / 0 / 2 | 8 / 0 |
| plain_context | +9.130 | 14 / 0 / 0 | 10 / 0 |
| majority_same_budget | +0.954 | 12 / 0 / 2 | 1 / 0 |
| provenance_grouped_vote | +1.337 | 12 / 0 / 2 | 1 / 0 |

## IDK Sufficiency Analysis

| Model | Without protection | Full v5.1 | Effect | Answerable | IDK-gold |
|---|---:|---:|---:|---:|---:|
| llama | 84.07 | 87.61 | +3.54 | 59/63 (93.65) | 40/50 (80.00) |
| qwen | 90.27 | 92.92 | +2.65 | 60/63 (95.24) | 45/50 (90.00) |
| mistral | 64.60 | 88.50 | +23.89 | 59/63 (93.65) | 41/50 (82.00) |

## Integrity

- Provenance replication: 17503 samples, 0 score changes, 0 winner changes.
- OpenBookQA was frozen before source generation with physical label isolation.
- The OpenBookQA result is negative relative to plain RAG and is retained, not tuned away.
- Post-hoc proof-completeness, contrastive-adjudication, and NLI-abstention candidates are diagnostic only and are not part of QPCF-v5.1.

## Claim Boundary

QPCF-v5.1 supports a conflict-focused claim on the published benchmark files and paired Golden/counterfactual contexts. It does not support universal improvement on partial-evidence, non-conflict QA: on OpenBookQA, QPCF trails matched plain RAG by 2.0 to 3.4 points.
