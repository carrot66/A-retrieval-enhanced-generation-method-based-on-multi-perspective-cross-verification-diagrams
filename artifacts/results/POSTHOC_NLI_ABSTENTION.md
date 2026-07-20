# Post-hoc NLI Abstention Audit

This is a diagnostic candidate evaluated after OpenBookQA outcomes were observed. It is not a frozen holdout result and does not replace the reported QPCF-v5.1 method.

| Model | Dataset | N | Frozen | Candidate | Plain | Candidate - frozen | Abstain rate | Rescue / harm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama | faitheval | 1000 | 83.80 | 83.20 | 70.50 | -0.60 | 37.70 | 1 / 7 |
| llama | musique_golden | 1772 | 92.72 | 92.27 | 91.14 | -0.45 | 79.06 | 10 / 18 |
| llama | squad_golden | 1769 | 98.42 | 98.36 | 97.85 | -0.06 | 40.31 | 1 / 2 |
| llama | timeqa | 113 | 87.61 | 87.61 | 81.42 | +0.00 | 72.57 | 0 / 0 |
| llama | musique_negative | 1772 | 88.43 | 87.53 | 77.14 | -0.90 | 72.57 | 3 / 19 |
| llama | squad_negative | 1769 | 88.86 | 88.69 | 75.52 | -0.17 | 46.13 | 0 / 3 |
| qwen | faitheval | 1000 | 76.70 | 76.40 | 66.60 | -0.30 | 37.70 | 2 / 5 |
| qwen | timeqa | 113 | 92.92 | 92.92 | 89.38 | +0.00 | 72.57 | 0 / 0 |
| qwen | musique_negative | 1772 | 86.74 | 86.79 | 72.74 | +0.06 | 72.57 | 16 / 15 |
| qwen | squad_negative | 1769 | 84.45 | 84.12 | 73.83 | -0.34 | 46.13 | 5 / 11 |
| mistral | faitheval | 1000 | 84.30 | 83.30 | 76.30 | -1.00 | 37.70 | 2 / 12 |
| mistral | timeqa | 113 | 88.50 | 89.38 | 70.80 | +0.88 | 72.57 | 1 / 0 |
| mistral | musique_negative | 1772 | 86.85 | 84.31 | 79.29 | -2.54 | 72.57 | 13 / 58 |
| mistral | squad_negative | 1769 | 87.17 | 86.09 | 75.35 | -1.07 | 46.13 | 7 / 26 |
| llama | openbookqa_holdout | 500 | 83.40 | 83.60 | 85.40 | +0.20 | 86.20 | 3 / 2 |
| llama | openbookqa_permuted | 500 | 83.20 | 83.80 | 86.60 | +0.60 | 86.20 | 6 / 3 |
| llama | openbookqa_distractor | 500 | 84.00 | 84.20 | 87.00 | +0.20 | 84.60 | 6 / 5 |
