# UGCP-v5.1-QPCF Paired Counterfactual Evaluation

Joint option accuracy requires the same method variant to answer both the Golden context and its answer-changing counterfactual context correctly.

| Pair | Variant | Golden | Counterfactual | Joint | Switch | Adapt given Golden | Stale given Golden |
|---|---|---:|---:|---:|---:|---:|---:|
| MuSiQue | full | 92.72 | 88.43 | 83.07 | 87.75 | 89.59 | 8.52 |
| MuSiQue | plain_context | 91.14 | 77.14 | 70.94 | 78.39 | 77.83 | 18.64 |
| MuSiQue | majority_same_budget | 91.42 | 87.58 | 80.81 | 86.74 | 88.40 | 9.69 |
| MuSiQue | provenance_grouped_vote | 91.31 | 86.79 | 79.97 | 86.23 | 87.58 | 10.38 |
| SQuAD | full | 98.42 | 88.75 | 87.56 | 89.49 | 88.97 | 9.82 |
| SQuAD | plain_context | 97.85 | 75.18 | 73.88 | 77.33 | 75.51 | 21.78 |
| SQuAD | majority_same_budget | 98.30 | 87.96 | 86.55 | 88.69 | 88.04 | 10.58 |
| SQuAD | provenance_grouped_vote | 98.25 | 86.49 | 85.02 | 87.22 | 86.54 | 12.08 |

## Paired Comparisons

| Pair | Comparator | Joint effect | 95% bootstrap CI | Holm p |
|---|---|---:|---:|---:|
| MuSiQue | without_query_semantic | +1.30 | [+0.56, +2.03] | 0.01566 |
| MuSiQue | without_local_semantic | +1.02 | [+0.34, +1.69] | 0.06293 |
| MuSiQue | without_semantic_bridge | +1.52 | [+0.73, +2.31] | 0.003961 |
| MuSiQue | without_provenance_capacity | +0.11 | [-0.11, +0.34] | 1 |
| MuSiQue | shared_backbone_capacity | +0.51 | [-0.23, +1.24] | 1 |
| MuSiQue | without_typed_topology | +0.06 | [-0.28, +0.40] | 1 |
| MuSiQue | without_directional_cut | +0.00 | [-0.28, +0.28] | 1 |
| MuSiQue | without_risk_projection | +1.86 | [+0.96, +2.77] | 0.001929 |
| MuSiQue | without_idk_protection | +0.00 | [+0.00, +0.00] | 1 |
| MuSiQue | scope_only | +0.23 | [-0.57, +1.02] | 1 |
| MuSiQue | plain_context | +12.13 | [+10.33, +14.00] | 1.153e-35 |
| MuSiQue | majority_same_budget | +2.26 | [+0.96, +3.61] | 0.01993 |
| MuSiQue | provenance_grouped_vote | +3.10 | [+1.69, +4.51] | 0.000419 |
| SQuAD | without_query_semantic | +0.34 | [-0.06, +0.73] | 1 |
| SQuAD | without_local_semantic | +0.28 | [-0.06, +0.62] | 1 |
| SQuAD | without_semantic_bridge | +0.96 | [+0.45, +1.53] | 0.01639 |
| SQuAD | without_provenance_capacity | +0.00 | [+0.00, +0.00] | 1 |
| SQuAD | shared_backbone_capacity | -0.40 | [-0.79, +0.00] | 1 |
| SQuAD | without_typed_topology | -0.11 | [-0.34, +0.11] | 1 |
| SQuAD | without_directional_cut | +0.00 | [-0.28, +0.28] | 1 |
| SQuAD | without_risk_projection | +1.70 | [+0.90, +2.49] | 0.0005488 |
| SQuAD | without_idk_protection | +0.00 | [+0.00, +0.00] | 1 |
| SQuAD | scope_only | +0.79 | [+0.23, +1.36] | 0.1881 |
| SQuAD | plain_context | +13.68 | [+11.98, +15.43] | 3.979e-54 |
| SQuAD | majority_same_budget | +1.02 | [-0.11, +2.15] | 1 |
| SQuAD | provenance_grouped_vote | +2.54 | [+1.30, +3.84] | 0.00281 |

## Evidence Boundary

- This is a secondary paired option-level analysis, not the published-paper ACC gate.
- Questions and choices are identical within each pair; contexts change and correct options are disjoint.
- Predictions come from separately frozen per-sample traces. Gold answers are used only after all option IDs are fixed.
- Only Llama is included because Qwen and Mistral were not run on Golden files under the formal protocol.
