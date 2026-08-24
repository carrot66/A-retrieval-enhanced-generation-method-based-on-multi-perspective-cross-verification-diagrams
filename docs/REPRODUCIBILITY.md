# Reproducibility

## Environment

- Python 3.10 or newer.
- Dependencies in `code/requirements.txt`.
- An OpenAI-compatible vLLM endpoint for source regeneration.
- Formal sample parallelism fixed at `workers=12`.

## Frozen Replay

From the release root:

```bash
PYTHON=python bash code/scripts/verify_release.sh
```

For separate execution, run the tests with `PYTHONPATH=code/src`, then invoke
`PYTHON=python bash code/scripts/replay_formal.sh` from the release root.

Replay consumes `artifacts/source/`, `artifacts/semantic/`, and `data/`. It writes
per-sample decision variants to `artifacts/projections/`, checks all configured point-estimate
gates, then writes Wilson intervals, exact McNemar tests, paired bootstrap intervals,
and Holm-adjusted results to `artifacts/results/`. It finally repeats every active
answer observation eight times within its original provenance family and verifies that
all maximum-flow support vectors and winners remain unchanged. This check writes
`artifacts/results/PROVENANCE_REPLICATION.json` and `.md`.

The same command then rebuilds `PAPER_REPORT.*`, runs a machine-readable release audit,
and regenerates `artifacts/manifests/SHA256SUMS` and `RELEASE.json`. The audit checks all
14 formal model-dataset pairs for complete sample indices, full-split source metadata,
raw model responses, semantic-artifact hashes, projection hashes, result gates, and
claim boundaries. Its outputs are `artifacts/results/RELEASE_AUDIT.json` and `.md`.

## Fresh Sufficiency Sensitivity Run

With a Llama OpenAI-compatible endpoint serving model name `llama` on port 8004:

```bash
PYTHON=/root/user/cs_tcci_fengzihuan/miniconda3/envs/vllm312/bin/python \
BASE_URL=http://127.0.0.1:8004/v1 \
MODEL=llama PROMPT_FORMAT=llama \
bash code/scripts/run_sufficiency_sensitivity.sh
```

This command physically separates every answer key before inference, regenerates
plain and graph observations with one sample per request, recomputes the two semantic
edges, and evaluates graph-certificate and IDK thresholds 2, 3, and 4. It writes raw
requests, raw responses, per-sample decisions, and logs under `artifacts/sensitivity/`.
The three thresholds share each newly generated observation so the paired comparison
changes only the decision threshold, not stochastic model outputs.

## Fresh Source-Channel Ablation Run

With the same Llama endpoint, run:

```bash
PYTHON=/root/user/cs_tcci_fengzihuan/miniconda3/envs/vllm312/bin/python \
BASE_URL=http://127.0.0.1:8004/v1 \
MODEL=llama PROMPT_FORMAT=llama \
bash code/scripts/run_source_channel_ablation.sh
```

The script refuses to reuse an existing `artifacts/source_ablation/` directory. It
physically separates all six answer-key files, regenerates every plain, graph, global
verifier, local verifier, local-semantic, and query-semantic observation, then fixes all
seven decisions (`full` plus one removal per source family) before joining labels. Raw
one-sample requests and responses, semantic records, per-sample decisions, and stage
logs are retained. Each removal also disables routing predicates directly produced by
that source family. Removing `graph_call` disables the context-versus-prior conflict
route because that family is its sole producer, leaving the best remaining source as
the deterministic fallback.

The integrity manifest deliberately excludes `artifacts/manifests/` itself and Python
bytecode caches. The verification entry point checks every included release file; the
standalone command is:

```bash
sha256sum -c artifacts/manifests/SHA256SUMS
```

Replay also aligns the Llama MuSiQue and SQuAD Golden/counterfactual files by index and
computes joint option accuracy after inference. The paired report verifies identical
questions and choices, changed contexts, and disjoint correct options before writing
`artifacts/results/PAIRED_COUNTERFACTUAL.json` and `.md`. This is a secondary robustness
analysis and does not replace the official evaluator.

The replication check can also be run independently:

```bash
PYTHONPATH=code/src python -m ugcp_v5.provenance_robustness \
  --config code/configs/experiment.json --root . --replicas 8
```

This is an observation-level implementation invariant. It does not test duplicated
retrieval documents, paraphrased passages, or independently generated model calls.
Gold fields are not accessed or used by the check.

The paired analysis can be run independently:

```bash
PYTHONPATH=code/src python -m ugcp_v5.paired_counterfactual \
  --config code/configs/experiment.json --root . \
  --bootstrap 10000 --seed 20260710
```

Pair predictions are fixed before answers are joined for option-level evaluation.

## Source Regeneration

The release preserves both source-stage implementations recorded by the formal trace
summaries. `source_runner_legacy.py` has SHA-256 `63b6f884...` and generated the Llama
and Qwen traces. `source_runner.py` has SHA-256 `d93613bb...` and generated the Mistral
traces. The latter adds label-isolated output support but keeps the same prompts and
decision logic. Both CLIs support `plain` and `graph` modes. A typical pair is:

```bash
python code/source_generation/source_runner.py \
  --dataset faitheval --mode plain \
  --base_url http://127.0.0.1:8003/v1 \
  --model llama --prompt_format llama --workers 12 \
  --output artifacts/source/llama/faitheval.plain.jsonl

python code/source_generation/source_runner.py \
  --dataset faitheval --mode graph \
  --base_input artifacts/source/llama/faitheval.plain.jsonl \
  --base_url http://127.0.0.1:8003/v1 \
  --model llama --prompt_format llama --workers 12 \
  --output artifacts/source/llama/faitheval.graph.jsonl
```

The source runner fixes temperature zero and deterministic seed derivation. Conditional
verifiers are routed only by model outputs and option structure, never by gold labels.
The included traces contain post-decision scores for audit convenience; tests verify
that changing the gold field cannot change UGCP-v5 decisions.

## Query-Aware Evidence Grounding

```bash
PYTHONPATH=code/src python -m ugcp_v5.semantic_bridge \
  --dataset_path data/faitheval_data.json \
  --dataset_name faitheval \
  --output artifacts/semantic/faitheval.bridge.jsonl \
  --workers 12 --local_device cpu --query_device cuda
```

The grounding stage records encoder names, package versions, device placement, dataset
hash, artifact hash, and confirms disabled gradients and no optimizer.

## Evaluation Integrity

`code/src/ugcp_v5/evaluate.py` is byte-identical to TruthfulRAG's evaluator with SHA-256
`ad2eafe2502f4f2a38855ab38b1a69f59165726e6586f79181e191dc5d506ebb`.
The four shared non-Golden data files are byte-identical to the TruthfulRAG repository.

## Frozen OpenBookQA Stress Test

The holdout protocol separates labels, freezes all code/data/model metadata hashes,
runs source generation, and joins answer keys only during replay:

```bash
PYTHON=/path/to/python-with-sentence-transformers \
PHASE=prepare MODEL=llama \
bash code/scripts/run_openbookqa_holdout.sh

$PYTHON code/data_preparation/freeze_holdout.py \
  --root . --model llama --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --output artifacts/holdout_protocol/HOLDOUT_FROZEN.json

PYTHON=$PYTHON PHASE=inference MODEL=llama \
BASE_URL=http://127.0.0.1:8005/v1 \
bash code/scripts/run_openbookqa_holdout.sh

PYTHON=$PYTHON PHASE=analyze MODEL=llama \
bash code/scripts/run_openbookqa_holdout.sh
```

The resulting trace audit checks 1,500 source pairs and 3,720 requests. Every source
record stores the exact request payload and raw model response. All six source summaries
report 500/500 completed samples, zero failures, `decision_only=true`, and
`full_official_evaluation=true`.

`artifacts/holdout_protocol/HOLDOUT_FROZEN.json` is the pre-inference snapshot. Files
under `code/diagnostics/` and `artifacts/development/` were created after holdout outcomes
were inspected; they are explicitly excluded from the frozen v5 method.
