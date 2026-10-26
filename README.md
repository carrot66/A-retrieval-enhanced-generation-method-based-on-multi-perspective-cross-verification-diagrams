# QPEF-RAG

This package contains the paper-facing implementation and the six evaluation
datasets for Query-aware Provenance-guided Evidence Fusion (QPEF-RAG). The
source tree retains the historical `ugcp_v5` path for script compatibility.

## Scope

The package includes:

- multi-view source generation for RAG evidence;
- query-aware semantic grounding;
- provenance-constrained graph fusion and answer selection;
- the six datasets used in the paper;
- the paper-facing aggregate result report in `results/`.

The package does not include model-generated request logs, per-sample traces,
or internal development and audit files. Those files are not required to read
or run the implementation and are kept out of the public code package.

## Results

The reported average conflict accuracy is 87.18% for Llama-3.1-8B-Instruct,
85.20% for Qwen2.5-7B-Instruct, and 86.70% for Mistral-7B-Instruct-v0.3.
The complete paper-facing table is available in:

- `results/PAPER_REPORT.md`
- `results/PAPER_REPORT.json`

## Layout

```text
code/
  configs/                 experiment configuration
  data_preparation/       label separation utility
  scripts/                replay helper
  source_generation/      model source generation and evaluator
  src/ugcp_v5/             grounding, graph fusion, projection, and replay
  tests/                   projection tests
data/                      six paper evaluation files
docs/                      method and reproducibility notes
results/                   aggregate paper-facing results
```

## Installation

```bash
python -m pip install -r code/requirements.txt
```

Python 3.10 or newer is recommended. Semantic grounding requires the
`sentence-transformers` models named in `code/src/ugcp_v5/semantic_bridge.py`.

## Run the tests

From the package root:

```bash
PYTHONPATH=code/src python -m unittest discover -s code/tests
```

## Regenerate source evidence

Source generation requires an OpenAI-compatible vLLM endpoint. A typical plain
and graph run is:

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

The generated files are runtime outputs and are intentionally not part of this
upload package. See `docs/REPRODUCIBILITY.md` for the full pipeline.
