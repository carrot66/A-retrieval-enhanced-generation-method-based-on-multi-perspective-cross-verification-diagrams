# MVCG-RAG: Multi-View Cross-Verification Graph RAG

This release contains the training-free, zero-shot MVCG-RAG implementation for
knowledge-conflict question answering. It uses an LLM to produce context and
prior observations, query-aware semantic verification to ground the observations,
and a provenance-constrained evidence graph to aggregate support without counting
multiple fields from one call as independent evidence.

## Formal Protocol

- Multiple-choice RAG evaluation with the unchanged TruthfulRAG evaluator.
- One label-isolated sample per model request; answer keys are unavailable during
  model inference and graph decisions.
- No training, fine-tuning, demonstrations, gold routing, or dataset-specific
  thresholds.
- `workers=12`, temperature `0`, and per-sample raw request/response retention.
- Llama runs all six datasets. Qwen and Mistral run the four non-Golden datasets
  in the main cross-model experiment.

## Method Modules

1. **Multi-view evidence generation:** obtain plain context reasoning, structured
   conflict analysis, and blind verification observations.
2. **Query-aware semantic grounding:** measure whether each option is supported by
   the question and the retrieved context.
3. **Provenance-constrained graph fusion:** connect source, observation, and option
   nodes; source capacities prevent repeated fields from one call being counted as
   separate support.
4. **Conservative decision:** update the fallback answer only when the graph gives
   unique, sufficient, and cross-source evidence; explicit `I don't know` options
   receive an additional sufficiency guard.

The graph is an evidence-fusion graph, not a factual knowledge graph and not an
assertion graph. The distinction between factual assertions and branch option
observations is documented in `docs/METHOD.md` and `docs/METHOD_ZH.md`.

## Results

The complete fresh Llama mechanism ablation contains 14 configurations over six
datasets: `artifacts/results/MECHANISM_ABLATION.json` and `.md`.

- Fresh Llama samples: `114,730`
- Retained model requests: `272,730`
- Failed requests: `0`
- Full MVCG-RAG weighted ACC: `90.93%`

Additional fresh experiments are retained in:

- `artifacts/source/`: real Llama, Qwen, and Mistral source traces.
- `artifacts/baselines/`: real baseline traces.
- `artifacts/source_ablation/`: one-source-at-a-time ablation traces.
- `artifacts/sensitivity/`: evidence-sufficiency threshold traces.
- `artifacts/mechanism_ablation/`: fresh per-variant Llama traces, semantic records,
  and final decisions.
- `artifacts/results/MAIN_COMPARISON.*`: requested external baseline table.
- `artifacts/results/SOURCE_CHANNEL_ABLATION.*`: source-level ablation results.
- `artifacts/results/SUFFICIENCY_SENSITIVITY.*`: threshold sensitivity results.

All result tables are accompanied by per-sample JSONL records and source hashes.
The release does not include historical decision replay artifacts.

## Reproduction

Install `code/requirements.txt`, start an OpenAI-compatible vLLM endpoint, and use
the scripts under `code/scripts/` with `workers=12`. Fresh formal runs are:

```bash
PYTHON=/path/to/python BASE_URL=http://127.0.0.1:8004/v1 \
  bash code/scripts/run_mechanism_ablation.sh
```

The existing raw traces are already the outputs of these per-sample model calls.
Run `bash code/scripts/verify_release.sh` to validate the retained result files,
label isolation, request counts, worker setting, and failure-free audits.
