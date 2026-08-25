# Reproducibility

## Environment

- Python 3.10 or newer.
- Dependencies in `code/requirements.txt`.
- An OpenAI-compatible vLLM endpoint.
- Formal sample parallelism fixed at `workers=12`.

## Label Isolation

`code/data_preparation/isolate_holdout_labels.py` creates two files for every
dataset: an inference file without answer fields and a separate answer-key file.
The source runner accepts only the inference file in decision-only mode. Gold
answers are joined only by the analysis programs after all model observations and
decisions have been written.

## Fresh Formal Runs

The primary fresh run is:

```bash
PYTHON=/root/user/cs_tcci_fengzihuan/miniconda3/envs/vllm312/bin/python \
BASE_URL=http://127.0.0.1:8004/v1 MODEL=llama PROMPT_FORMAT=llama \
bash code/scripts/run_mechanism_ablation.sh
```

It uses 14 configurations and six datasets. Each configuration regenerates its
plain and graph observations with one label-isolated sample per request, then
computes semantic grounding and final decisions. The run refuses to reuse an
existing output root so that a fresh experiment cannot silently mix old traces.
For an interrupted run, `code/scripts/resume_mechanism_ablation.sh` continues only
missing sample indices and preserves completed raw records.

The other formal fresh runs are:

```bash
bash code/scripts/run_source_channel_ablation.sh
bash code/scripts/run_sufficiency_sensitivity.sh
```

Both scripts retain raw requests, raw responses, semantic records, per-sample
decisions, hashes, and stage logs.

## Verification

```bash
bash code/scripts/verify_release.sh
```

The verifier checks the current formal result JSON files, fresh-inference flags,
zero failures, answer-key isolation, one-sample request protocol, and `workers=12`.
It does not regenerate decisions from historical observations.

## Evaluation

`code/source_generation/evaluate.py` is the TruthfulRAG-compatible evaluator used
for every reported accuracy. The six release datasets are stored under `data/`.
All per-sample results remain available under `artifacts/` for audit and paper
reproduction.
