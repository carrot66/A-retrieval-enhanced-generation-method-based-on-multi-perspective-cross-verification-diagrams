# Fresh Mechanism Ablation

Every configuration below was generated from a separate, fresh Llama run. Each request contains exactly one label-isolated sample; no historical source trace or decision replay is read.

| Variant | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC | Delta vs. full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full MVCG-RAG | 83.20 | 86.73 | 92.61 | 98.53 | 88.43 | 88.81 | 90.93 | +0.00 |
| w/o without_query_semantic | 81.00 | 86.73 | 91.93 | 98.47 | 87.64 | 88.52 | 90.27 | -0.66 |
| w/o without_local_semantic | 81.80 | 86.73 | 92.27 | 98.36 | 87.53 | 88.81 | 90.46 | -0.48 |
| w/o without_semantic_bridge | 80.60 | 87.61 | 92.10 | 98.42 | 87.25 | 87.96 | 90.05 | -0.88 |
| w/o without_provenance_capacity | 83.50 | 87.61 | 92.33 | 98.47 | 88.32 | 88.92 | 90.91 | -0.02 |
| w/o shared_backbone_capacity | 83.90 | 86.73 | 91.65 | 98.36 | 88.60 | 89.49 | 90.96 | +0.02 |
| w/o without_typed_topology | 82.50 | 87.61 | 92.44 | 98.64 | 88.21 | 88.81 | 90.80 | -0.13 |
| w/o without_directional_cut | 81.70 | 87.61 | 92.66 | 98.64 | 88.21 | 88.81 | 90.75 | -0.18 |
| w/o without_risk_projection | 80.30 | 81.42 | 91.99 | 98.76 | 87.08 | 87.17 | 89.77 | -1.16 |
| w/o without_idk_protection | 83.70 | 82.30 | 92.66 | 98.53 | 88.26 | 88.86 | 90.92 | -0.01 |
| w/o scope_only | 80.00 | 83.19 | 92.78 | 98.47 | 87.64 | 88.19 | 90.21 | -0.72 |
| w/o plain_context | 70.40 | 81.42 | 90.97 | 97.85 | 76.81 | 75.75 | 83.47 | -7.47 |
| w/o majority_same_budget | 77.10 | 78.76 | 91.08 | 98.42 | 87.64 | 88.07 | 89.40 | -1.54 |
| w/o provenance_grouped_vote | 75.70 | 77.88 | 90.86 | 98.25 | 86.96 | 86.66 | 88.68 | -2.26 |

## Trace Audit

- Fresh per-variant decisions: 114730
- Fresh retained Llama requests: 272730
- Source failures: 0
- Historical observation replay: no
- Gold labels available during model inference or decision: no
- Official evaluator SHA-256: `ad2eafe2502f4f2a38855ab38b1a69f59165726e6586f79181e191dc5d506ebb`
