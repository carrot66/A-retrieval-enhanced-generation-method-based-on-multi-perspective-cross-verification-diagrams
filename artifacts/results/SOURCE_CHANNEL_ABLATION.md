# Source-Channel Ablation

Every source trace in this report was regenerated sample by sample with the deployed model. Answer keys were physically separated before inference and joined only after all variant option IDs were fixed.

Each row removes exactly one source family from graph fusion, source-family support counting, and any routing predicate directly produced by that family. `w/o graph_call` disables the conflict route because this family is the only producer of the context-versus-prior conflict observation; it falls back to the best remaining source observation.

| Removed source channel | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC | Delta vs. full | Changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full MVCG-RAG | 83.40 | 87.61 | 92.78 | 98.42 | 88.49 | 88.75 | 90.98 | +0.00 | 0 |
| w/o plain | 78.00 | 77.88 | 88.77 | 97.46 | 88.88 | 91.41 | 89.77 | -1.21 | 552 |
| w/o graph_call | 70.30 | 81.42 | 90.97 | 97.85 | 76.81 | 75.64 | 83.43 | -7.55 | 875 |
| w/o global_verifier | 82.40 | 90.27 | 92.10 | 98.36 | 88.15 | 88.98 | 90.71 | -0.27 | 85 |
| w/o local_verifier | 83.60 | 88.50 | 92.61 | 98.42 | 88.21 | 88.69 | 90.91 | -0.07 | 49 |
| w/o local_semantic | 81.90 | 88.50 | 92.33 | 98.30 | 87.81 | 88.64 | 90.52 | -0.46 | 75 |
| w/o query_semantic | 81.10 | 87.61 | 92.16 | 98.42 | 87.47 | 88.47 | 90.29 | -0.70 | 103 |

## Paired Tests

| Removed source channel | Full better / ablation better | Effect (pp) | 95% bootstrap CI | McNemar p (Holm) |
|---|---:|---:|---:|---:|
| w/o plain | 295/196 | +1.21 | [+0.70, +1.74] | 9.13e-06 (2.739e-05) |
| w/o graph_call | 726/107 | +7.55 | [+6.89, +8.25] | 8.73e-114 (5.238e-113) |
| w/o global_verifier | 50/28 | +0.27 | [+0.06, +0.48] | 0.01685 (0.03371) |
| w/o local_verifier | 24/18 | +0.07 | [-0.07, +0.23] | 0.4408 (0.4408) |
| w/o local_semantic | 54/16 | +0.46 | [+0.27, +0.67] | 5.854e-06 (2.342e-05) |
| w/o query_semantic | 77/20 | +0.70 | [+0.46, +0.94] | 4.591e-09 (2.295e-08) |

## Trace Audit

- Fresh per-sample decisions: 8195
- Fresh retained model requests: 19436
- Source failures: 0
- Historical observation replay: no
- Gold labels available during model inference or decision: no
- Official evaluator SHA-256: `ad2eafe2502f4f2a38855ab38b1a69f59165726e6586f79181e191dc5d506ebb`
