# Main Comparison Baselines

The main comparison follows Table 1 of `2506.08938v2.pdf` and uses the four conflict datasets: FaithEval, RealtimeQA, MuSiQue-Negative, and SQuAD-Negative.

## Method Scope

- `w/o RAG` is the deployed backbone with the question and choices but no retrieved context.
- `RAG` is the same backbone with the complete retrieved context.
- `KRE` is the published dependent-role intervention: the prompt asks the model to rely only on the supplied context. The local runner uses this role in a zero-shot setting and writes one record per sample.
- `CAD` and `COIECD` alter token-level decoding distributions. Their published values are included from FaithfulRAG Table 1; they are not labelled local reproductions without token-level probability traces.
- `Self-RAG` requires the separately trained Self-RAG checkpoint. Its published Llama2-7B row is retained with its original backbone label and is not treated as a same-backbone comparison.

## Local Rerun

Start one deployed model at a time, then run:

```bash
MODELS=llama RUN_ROOT=/root/user/cs_tcci_fengzihuan/我的实验/final/UGCP-v5-QPCF \
  PYTHON=python bash code/scripts/run_baselines.sh
```

The default endpoints are Llama `8004`, Qwen `8003`, and Mistral `8002`. Override them with `LLAMA_BASE_URL`, `QWEN_BASE_URL`, or `MISTRAL_BASE_URL` when needed. The runner uses 12 workers, temperature 0, no training, no demonstrations, and never sends the gold answer in a prompt.

The output is written to `artifacts/baselines/<model>/<dataset>/` as JSONL plus a summary. `MAIN_COMPARISON.md` and the unified `PAPER_REPORT.md` are generated under `artifacts/results/`. A local row replaces the corresponding published row only after all four datasets for that model and method are present.

Published rows are provenance-labelled rather than presented as new local model calls. A point estimate from another paper is not a paired per-sample reproduction.
