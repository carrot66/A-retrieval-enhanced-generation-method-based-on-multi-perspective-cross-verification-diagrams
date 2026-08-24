# Main Comparison

ACC (%) under the same TruthfulRAG official evaluator and the paper's four conflict datasets.

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

Published rows are reported values from FaithfulRAG Table 1, not silently relabeled local reruns.
Local w/o RAG, RAG, and KRE rows replace the corresponding published row only when complete per-sample outputs exist.
Self-RAG is reported with its original Llama2-7B backbone and is not a same-backbone comparison to MVCG-RAG.
CAD and COIECD require token-level probability decoding; their published rows remain separate until an exact decoder run is available.
