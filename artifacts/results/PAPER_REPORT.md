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

## Main Baseline Comparison

| Method | Backbone | FaithEval | RealtimeQA | MuSiQue | SQuAD | Source |
|---|---|---:|---:|---:|---:|---|
| w/o RAG | Llama3.1-8B-Instruct | 7.30 | 46.02 | 14.28 | 10.01 | local zero-shot rerun |
| w/o RAG | Qwen2.5-7B-Instruct | 4.20 | 40.70 | 19.60 | 11.10 | FaithfulRAG Table 1 |
| w/o RAG | Mistral-7B-Instruct | 6.30 | 29.20 | 13.80 | 11.50 | FaithfulRAG Table 1 |
| RAG | Llama3.1-8B-Instruct | 66.60 | 84.07 | 64.45 | 68.12 | local zero-shot rerun |
| RAG | Qwen2.5-7B-Instruct | 53.10 | 78.70 | 75.20 | 68.30 | FaithfulRAG Table 1 |
| RAG | Mistral-7B-Instruct | 61.90 | 52.20 | 67.60 | 67.20 | FaithfulRAG Table 1 |
| Self-RAG | Llama2-7B | 37.40 | 55.80 | 54.10 | 62.00 | FaithfulRAG Table 1 |
| KRE | Llama3.1-8B-Instruct | 66.50 | 95.58 | 68.23 | 73.71 | local zero-shot rerun |
| KRE | Qwen2.5-7B-Instruct | 59.60 | 86.70 | 70.70 | 73.70 | FaithfulRAG Table 1 |
| KRE | Mistral-7B-Instruct | 73.20 | 76.90 | 50.60 | 74.60 | FaithfulRAG Table 1 |
| CAD | Llama3.1-8B-Instruct | 66.20 | 61.90 | 72.60 | 71.20 | FaithfulRAG Table 1 |
| CAD | Qwen2.5-7B-Instruct | 60.50 | 77.00 | 78.60 | 73.40 | FaithfulRAG Table 1 |
| CAD | Mistral-7B-Instruct | 60.20 | 55.80 | 63.60 | 66.90 | FaithfulRAG Table 1 |
| COIECD | Llama3.1-8B-Instruct | 67.70 | 62.80 | 70.50 | 71.80 | FaithfulRAG Table 1 |
| COIECD | Qwen2.5-7B-Instruct | 62.30 | 78.80 | 69.70 | 70.80 | FaithfulRAG Table 1 |
| COIECD | Mistral-7B-Instruct | 62.80 | 58.40 | 66.80 | 65.40 | FaithfulRAG Table 1 |
| MVCG-RAG (Ours) | Meta-Llama-3.1-8B-Instruct | 83.80 | 87.61 | 88.43 | 88.86 | local MVCG-RAG result |
| MVCG-RAG (Ours) | Qwen2.5-7B-Instruct | 76.70 | 92.92 | 86.74 | 84.45 | local MVCG-RAG result |
| MVCG-RAG (Ours) | mistralai/Mistral-7B-Instruct-v0.3 | 84.30 | 88.50 | 86.85 | 87.17 | local MVCG-RAG result |

The w/o RAG, RAG, and KRE rows are local zero-shot reruns when complete per-sample outputs are present; otherwise they retain the FaithfulRAG Table 1 point estimate.
Self-RAG retains its original Llama2-7B backbone label. CAD and COIECD retain published values unless exact token-level decoder traces are supplied.

## External Standard-RAG Reference

The common external reference is the local Llama standard-RAG result. It is used only for the external aggregate comparison and is not substituted into module ablations or same-observation controls.

| Reference | Backbone | FaithEval | RealtimeQA | MuSiQue-Negative | SQuAD-Negative | Average ACC |
|---|---|---:|---:|---:|---:|---:|
| Standard RAG | Llama3.1-8B-Instruct | 66.60 | 84.07 | 64.45 | 68.12 | 70.81 |

External-reference scope: faitheval, timeqa, musique_negative, squad_negative.

## Internal Same-Observation Controls

The following values reuse frozen per-sample observations. Plain-context replay, majority voting, and provenance-grouped voting are internal controls for decision analysis, not the external standard-RAG baseline.

| Model | Full QPCF | Plain-context replay | Majority | Grouped vote | Full - Plain replay | Full - Majority | Full - Grouped |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama | 87.58 | 75.20 | 85.28 | 84.23 | +12.38 | +2.30 | +3.35 |
| qwen | 83.86 | 72.24 | 83.97 | 83.71 | +11.62 | -0.11 | +0.15 |
| mistral | 86.46 | 76.94 | 85.60 | 85.54 | +9.52 | +0.86 | +0.92 |

Internal-control scope: faitheval, timeqa, musique_negative, squad_negative. Golden datasets remain in the primary and paired analyses.

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
| without_query_semantic | +1.590 | 9 / 3 / 0 | 4 / 0 |
| without_local_semantic | +1.497 | 11 / 1 / 0 | 5 / 0 |
| without_semantic_bridge | +3.617 | 11 / 1 / 0 | 9 / 0 |
| without_provenance_capacity | +0.079 | 5 / 6 / 1 | 0 / 0 |
| shared_backbone_capacity | -0.680 | 2 / 3 / 7 | 0 / 3 |
| without_typed_topology | +0.244 | 7 / 3 / 2 | 0 / 0 |
| without_directional_cut | +0.236 | 7 / 4 / 1 | 1 / 0 |
| without_risk_projection | +1.031 | 11 / 0 / 1 | 2 / 0 |
| without_idk_protection | +0.244 | 3 / 9 / 0 | 1 / 0 |
| scope_only | +4.054 | 12 / 0 / 0 | 8 / 0 |

## IDK Sufficiency Analysis

| Model | Without protection | Full v5.1 | Effect | Answerable | IDK-gold |
|---|---:|---:|---:|---:|---:|
| llama | 84.07 | 87.61 | +3.54 | 59/63 (93.65) | 40/50 (80.00) |
| qwen | 90.27 | 92.92 | +2.65 | 60/63 (95.24) | 45/50 (90.00) |
| mistral | 64.60 | 88.50 | +23.89 | 59/63 (93.65) | 41/50 (82.00) |

## Evidence Sufficiency Threshold Sensitivity

Every observation in this analysis was regenerated with fresh, label-isolated, one-sample model requests. The graph and IDK thresholds are varied in separate single-factor panels.

| Graph minimum support | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 83.70 | 87.61 | 92.72 | 98.53 | 88.37 | 88.86 | 91.03 |
| 3 | 83.40 | 87.61 | 92.78 | 98.53 | 88.21 | 88.86 | 90.97 |
| 4 | 81.60 | 87.61 | 92.72 | 98.59 | 87.87 | 88.47 | 90.59 |

| IDK minimum source families | RealtimeQA ACC | Changed vs. default 3 |
|---:|---:|---:|
| 2 | 83.19 | 7 |
| 3 | 87.61 | 0 |
| 4 | 89.38 | 2 |

Trace audit: 8195 fresh samples, 19454 retained model requests, 0 failures.

## Complete Source-Channel Ablation

All six source families are removed one at a time on newly generated, label-isolated Llama traces. Each removal excludes the family from fusion, sufficiency counting, and its source-specific routing predicates.

| Variant | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC | Delta vs. full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 83.40 | 87.61 | 92.78 | 98.42 | 88.49 | 88.75 | 90.98 | +0.00 |
| w/o plain | 78.00 | 77.88 | 88.77 | 97.46 | 88.88 | 91.41 | 89.77 | -1.21 |
| w/o graph_call | 70.30 | 81.42 | 90.97 | 97.85 | 76.81 | 75.64 | 83.43 | -7.55 |
| w/o global_verifier | 82.40 | 90.27 | 92.10 | 98.36 | 88.15 | 88.98 | 90.71 | -0.27 |
| w/o local_verifier | 83.60 | 88.50 | 92.61 | 98.42 | 88.21 | 88.69 | 90.91 | -0.07 |
| w/o local_semantic | 81.90 | 88.50 | 92.33 | 98.30 | 87.81 | 88.64 | 90.52 | -0.46 |
| w/o query_semantic | 81.10 | 87.61 | 92.16 | 98.42 | 87.47 | 88.47 | 90.29 | -0.70 |

Trace audit: 8195 fresh samples, 19436 retained model requests, 0 failures.

## Integrity

- Provenance replication: 17503 samples, 0 score changes, 0 winner changes.
- OpenBookQA was frozen before source generation with physical label isolation.
- The OpenBookQA result is negative relative to plain RAG and is retained, not tuned away.
- Post-hoc proof-completeness, contrastive-adjudication, and NLI-abstention candidates are diagnostic only and are not part of QPCF-v5.1.

## Claim Boundary

QPCF-v5.1 supports a conflict-focused claim on the published benchmark files and paired Golden/counterfactual contexts. It does not support universal improvement on partial-evidence, non-conflict QA: on OpenBookQA, QPCF trails matched plain RAG by 2.0 to 3.4 points.
