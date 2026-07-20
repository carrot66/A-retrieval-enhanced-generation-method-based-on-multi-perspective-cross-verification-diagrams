# UGCP-v5.1-QPCF Provenance Replication Check

Each active answer observation was repeated within its original source family. The source-to-family edge retained capacity one.

| Model | Dataset | N | Reconstruction mismatches | Score changes | Winner changes |
|---|---|---:|---:|---:|---:|
| llama | faitheval | 1000 | 0 | 0 | 0 |
| llama | musique_golden | 1772 | 0 | 0 | 0 |
| llama | squad_golden | 1769 | 0 | 0 | 0 |
| llama | timeqa | 113 | 0 | 0 | 0 |
| llama | musique_negative | 1772 | 0 | 0 | 0 |
| llama | squad_negative | 1769 | 0 | 0 | 0 |
| qwen | faitheval | 1000 | 0 | 0 | 0 |
| qwen | timeqa | 113 | 0 | 0 | 0 |
| qwen | musique_negative | 1772 | 0 | 0 | 0 |
| qwen | squad_negative | 1769 | 0 | 0 | 0 |
| mistral | faitheval | 1000 | 0 | 0 | 0 |
| mistral | timeqa | 113 | 0 | 0 | 0 |
| mistral | musique_negative | 1772 | 0 | 0 | 0 |
| mistral | squad_negative | 1769 | 0 | 0 | 0 |

Status: **PASS** over 17503 samples with 8 added replicas per active observation.

This is an observation-level implementation check. It does not establish invariance to duplicated retrieved documents, paraphrased evidence, or newly generated model calls.
