# UGCP-v5.1-QPCF Frozen Holdout Results (llama)

## Accuracy

| Variant | N | Full ACC | 95% Wilson CI | Majority ACC | Full - majority |
|---|---:|---:|---:|---:|---:|
| base | 500 | 83.40 | [79.89, 86.40] | 84.00 | -0.60 |
| permuted | 500 | 83.20 | [79.67, 86.22] | 81.80 | +1.40 |
| distractor | 500 | 84.00 | [80.53, 86.95] | 83.60 | +0.40 |

## End-to-End Robustness

| Perturbation | Full prediction invariance | Majority invariance | Full ACC change |
|---|---:|---:|---:|
| permuted | 85.80 | 86.00 | -0.20 |
| distractor | 92.00 | 92.60 | +0.60 |

## Intervention Diagnostics

| Variant | Plain ACC | Full ACC | Changed | Rescued | Harmed | Net rescues |
|---|---:|---:|---:|---:|---:|---:|
| base | 85.40 | 83.40 | 40 | 10 | 20 | -10 |
| permuted | 86.60 | 83.20 | 43 | 10 | 27 | -17 |
| distractor | 87.00 | 84.00 | 35 | 5 | 20 | -15 |

## Trace Completeness

| Variant | Records | Requests | Conditional verifier pairs | Missing | Label fields |
|---|---:|---:|---:|---:|---:|
| base | 500 | 1236 | 118 | 0 | 0 |
| permuted | 500 | 1236 | 118 | 0 | 0 |
| distractor | 500 | 1248 | 124 | 0 | 0 |

## Evidence Boundary

- The method and perturbation generators were frozen before model outputs were produced.
- Source generation is decision-only; gold labels are joined during replay after option IDs are fixed.
- McNemar and bootstrap claims are internal paired comparisons, not comparisons against published paper rows.
- No model or dataset-specific threshold is introduced for this holdout.
