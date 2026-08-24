# UGCP-v5.1-QPCF Development Results

## Primary Results

| Model | Dataset | N | ACC | 95% CI | Target | Margin | Pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| llama | FaithEval | 1000 | 83.80 | [81.39, 85.95] | 79.80 | +4.00 | yes |
| llama | MuSiQue-Golden | 1772 | 92.72 | [91.42, 93.84] | 85.30 | +7.42 | yes |
| llama | SQuAD-Golden | 1769 | 98.42 | [97.72, 98.90] | 96.60 | +1.82 | yes |
| llama | RealtimeQA | 113 | 87.61 | [80.27, 92.47] | 81.40 | +6.21 | yes |
| llama | MuSiQue-Negative | 1772 | 88.43 | [86.86, 89.84] | 79.90 | +8.53 | yes |
| llama | SQuAD-Negative | 1769 | 88.86 | [87.31, 90.25] | 86.30 | +2.56 | yes |
| qwen | FaithEval | 1000 | 76.70 | [73.98, 79.21] | 73.20 | +3.50 | yes |
| qwen | RealtimeQA | 113 | 92.92 | [86.65, 96.37] | 82.30 | +10.62 | yes |
| qwen | MuSiQue-Negative | 1772 | 86.74 | [85.08, 88.24] | 79.10 | +7.64 | yes |
| qwen | SQuAD-Negative | 1769 | 84.45 | [82.69, 86.07] | 78.70 | +5.75 | yes |
| mistral | FaithEval | 1000 | 84.30 | [81.91, 86.42] | 81.90 | +2.40 | yes |
| mistral | RealtimeQA | 113 | 88.50 | [81.31, 93.15] | 81.40 | +7.10 | yes |
| mistral | MuSiQue-Negative | 1772 | 86.85 | [85.20, 88.35] | 79.30 | +7.55 | yes |
| mistral | SQuAD-Negative | 1769 | 87.17 | [85.53, 88.65] | 85.70 | +1.47 | yes |

## Same-Budget Majority

The same four non-Golden conflict datasets are used for every baseline and ablation aggregate. `plain_context` is the unified standard-RAG baseline; majority and provenance-grouped vote are budget-matched controls.

| Model | Full ACC | Standard RAG | Majority | Grouped vote | Full - RAG | Full - Majority | Full - Grouped |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama | 87.58 | 75.20 | 85.28 | 84.23 | +12.38 | +2.30 | +3.35 |
| qwen | 83.86 | 72.24 | 83.97 | 83.71 | +11.62 | -0.11 | +0.15 |
| mistral | 86.46 | 76.94 | 85.60 | 85.54 | +9.52 | +0.86 | +0.92 |

## Comparator Summary

| Comparator | Weighted effect | Positive/Tied/Negative pairs | Holm-significant positive/negative |
|---|---:|---:|---:|
| without_query_semantic | +1.590 | 9/3/0 | 4/0 |
| without_local_semantic | +1.497 | 11/1/0 | 5/0 |
| without_semantic_bridge | +3.617 | 11/1/0 | 9/0 |
| without_provenance_capacity | +0.079 | 5/6/1 | 0/0 |
| shared_backbone_capacity | -0.680 | 2/3/7 | 0/3 |
| without_typed_topology | +0.244 | 7/3/2 | 0/0 |
| without_directional_cut | +0.236 | 7/4/1 | 1/0 |
| without_risk_projection | +1.031 | 11/0/1 | 2/0 |
| without_idk_protection | +0.244 | 3/9/0 | 1/0 |
| scope_only | +4.054 | 12/0/0 | 8/0 |
| plain_context | +11.173 | 12/0/0 | 10/0 |
| majority_same_budget | +1.017 | 10/0/2 | 1/0 |
| provenance_grouped_vote | +1.475 | 10/0/2 | 2/0 |

## Evidence Boundary

- These are adaptive development-trace replays, not an untouched holdout.
- Majority and every ablation reuse exactly the same frozen model observations.
- Gold labels are read only after every variant option ID is fixed.
- Published targets are point estimates; paired significance against the papers requires reproduced per-sample baseline predictions.
- Unified baseline scope: faitheval, timeqa, musique_negative, squad_negative; Golden datasets remain in the primary and paired analyses but are excluded from these baseline and ablation aggregates.
