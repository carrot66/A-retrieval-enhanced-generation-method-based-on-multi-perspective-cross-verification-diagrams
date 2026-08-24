# Evidence Sufficiency Threshold Sensitivity

All observations in this analysis were regenerated sample by sample with the deployed model. Historical source traces were not used. Each API request contains exactly one sample; workers=12 controls only concurrent requests.

## Graph Certificate Threshold

The IDK protection threshold remains fixed at its formal value of 3, so this panel changes only the graph-certificate minimum support.

| Minimum support | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 83.70 | 87.61 | 92.72 | 98.53 | 88.37 | 88.86 | 91.03 |
| 3 | 83.40 | 87.61 | 92.78 | 98.53 | 88.21 | 88.86 | 90.97 |
| 4 | 81.60 | 87.61 | 92.72 | 98.59 | 87.87 | 88.47 | 90.59 |

## IDK Protection Threshold

The graph-certificate threshold remains fixed at its formal value of 2. This panel is restricted to RealtimeQA because the other formal datasets have no explicit abstention option.

| Minimum source families | RealtimeQA ACC | Changed vs. 3 |
|---:|---:|---:|
| 2 | 83.19 | 7 |
| 3 | 87.61 | 0 |
| 4 | 89.38 | 2 |

## Trace Audit

- Fresh per-sample decisions: 8195
- Fresh retained model requests: 19454
- Source failures: 0
- Gold labels available to model inference: no
- Official evaluator SHA-256: `ad2eafe2502f4f2a38855ab38b1a69f59165726e6586f79181e191dc5d506ebb`
