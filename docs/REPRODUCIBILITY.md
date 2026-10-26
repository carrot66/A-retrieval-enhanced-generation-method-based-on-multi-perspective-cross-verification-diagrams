# Reproducibility

## Environment

- Python 3.10 or newer.
- Dependencies in `code/requirements.txt`.
- An OpenAI-compatible vLLM endpoint for source generation.
- Twelve workers for the formal runs.

## Tests

From the package root:

```bash
PYTHONPATH=code/src python -m unittest discover -s code/tests
```

## Data and labels

The six JSON files in `data/` are the evaluation inputs used by the paper. The
source-generation code receives question, context, and choices. The label
separation utility can create an inference file without answer fields and a
separate answer-key file:

```bash
python code/data_preparation/isolate_labels.py \
  --source data/faitheval_data.json \
  --inference_output /tmp/faitheval.inference.json \
  --answer_key_output /tmp/faitheval.answer_key.json \
  --metadata_output /tmp/faitheval.isolation.json
```

Gold answers are joined only after the final option ID has been fixed.

## Pipeline

1. Generate plain and graph source records with
   `code/source_generation/source_runner.py` against an OpenAI-compatible
   endpoint.
2. Generate semantic grounding records with
   `python -m ugcp_v5.semantic_bridge`.
3. Replay the source and semantic records with
   `code/scripts/replay_formal.sh`.

The replay script writes runtime outputs under `artifacts/`. That directory is
created during execution and is intentionally excluded from this code upload
package. The committed files under `results/` are the paper-facing aggregate
results corresponding to the released experiment.

## Evaluation

Dataset-level accuracy uses the evaluator in
`code/source_generation/evaluate.py`. The graph stores source-process
observations and answer options. It applies a channel-level capacity constraint
so multiple fields from one generation process cannot be counted as separate
support.
