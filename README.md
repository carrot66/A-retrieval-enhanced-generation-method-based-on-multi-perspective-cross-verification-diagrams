# QPCF: Query-Aware Provenance-Constrained Fusion

QPCF is a training-free, zero-shot evidence-fusion method for multiple-choice RAG
when retrieved evidence conflicts with an LLM's parametric knowledge.
`UGCP-v5.1-QPCF` is the release identifier retained by the code and result files.
The method combines complementary LLM and semantic evidence while preventing
multiple outputs from one source from being counted as separate support.

## Method

The method has four reader-facing modules:

1. **Multi-view RAG evidence generation:** collect a standard context answer, a
   structured context-versus-prior reasoning view, and candidate-blind verification.
2. **Query-aware evidence grounding:** combine dense context matching with a
   question-conditioned NLI scorer.
3. **Provenance-aware graph evidence fusion:** aggregate source-distinct support while
   preventing one model call from contributing several votes.
4. **Conservative conflict-aware answer selection:** update the fallback answer only
   when one option has unique multi-source support. If an explicit `I don't know`
   option exists, require three provenance-distinct source families before emitting a
   concrete answer; otherwise abstain.

In implementation, source-family-disjoint support is measured with maximum flow on a directed
evidence graph. This is a source-aware aggregation mechanism, not a claim that QPCF
constructs a general factual knowledge graph. See `docs/METHOD.md` for the exact graph,
decision rule, ablations, and a mapping from code terms to paper-facing terminology.
The same explanation is available in Chinese in `docs/METHOD_ZH.md`.
The July 2026 literature comparison and claim boundary are recorded in
`docs/NOVELTY_AUDIT.md`.
Executable guarantees and their graph-theoretic limits are stated in
`docs/FORMAL_PROPERTIES.md`.

QPCF also distinguishes factual assertions from answer observations. A factual
assertion is a context-grounded, truth-evaluable proposition; a branch's option ID is
only an observation of that branch. The current graph has no factual assertion nodes
and is therefore not presented as an assertion graph.

Source-family distinction is operational, not statistical independence. Several LLM
branches share the same backbone; QPCF guarantees only that fields emitted by one call
cannot be counted repeatedly.

There is no task training, fine-tuning, demonstration, gold routing,
model-specific threshold, or dataset-specific threshold. Zero-shot does not mean
zero model calls: source generation uses two unconditional LLM calls and up to two
candidate-blind verification calls when the graph route requires them. Formal runs
use `workers=12` and temperature zero.

## Results

All values below use the unchanged TruthfulRAG evaluator. Published paper values
are descriptive gates, not paired baselines.

The paper-style main comparison, including w/o RAG, RAG, KRE, CAD, COIECD, and
Self-RAG, is in `artifacts/results/MAIN_COMPARISON.md`. Local zero-shot reruns
are stored under `artifacts/baselines/`; published rows remain explicitly labelled
when an exact local decoder or checkpoint is unavailable.

| Model | FaithEval | RealtimeQA | MuSiQue-Neg | SQuAD-Neg | MuSiQue-Gold | SQuAD-Gold |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 83.80 | 87.61 | 88.43 | 88.86 | 92.72 | 98.42 |
| Qwen2.5-7B | 76.70 | 92.92 | 86.74 | 84.45 | - | - |
| Mistral-7B-v0.3 | 84.30 | 88.50 | 86.85 | 87.17 | - | - |

The Mistral rows use the exact `mistralai/Mistral-7B-Instruct-v0.3` checkpoint named
by TruthfulRAG, with physically label-isolated inputs and 14,398 retained real GPU
requests. Ten plain calls returned empty completion text; these outputs were retained
without selective retry. All four exact-checkpoint gates pass.

The IDK sufficiency ablation isolates the v5.1 change on RealtimeQA. The other 13
formal model-dataset pairs contain no explicit abstention option and are bit-for-bit
unchanged in selected option IDs.

| Model | Without IDK protection | Full v5.1 | Answerable subset | IDK-gold subset |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 84.07 | 87.61 | 59/63 (93.65) | 40/50 (80.00) |
| Qwen2.5-7B | 90.27 | 92.92 | 60/63 (95.24) | 45/50 (90.00) |
| Mistral-7B-v0.3 | 64.60 | 88.50 | 59/63 (93.65) | 41/50 (82.00) |

### Evidence sufficiency sensitivity

A separate label-isolated Llama run regenerated all 8,195 formal samples with
19,454 retained one-sample model requests and zero failures. Historical model
observations were not reused. Varying only the graph-certificate minimum support
shows that the frozen value of 2 has the highest sample-weighted ACC.

| Graph minimum support | FaithEval | RealtimeQA | MuSiQue-Gold | SQuAD-Gold | MuSiQue-Neg | SQuAD-Neg | Weighted ACC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 83.70 | 87.61 | 92.72 | 98.53 | 88.37 | 88.86 | 91.03 |
| 3 | 83.40 | 87.61 | 92.78 | 98.53 | 88.21 | 88.86 | 90.97 |
| 4 | 81.60 | 87.61 | 92.72 | 98.59 | 87.87 | 88.47 | 90.59 |

The IDK threshold is varied separately on RealtimeQA while the graph threshold
remains fixed at 2: thresholds 2, 3, and 4 obtain 83.19, 87.61, and 89.38 ACC.
The value 4 is not adopted because this comparison uses test labels and therefore
cannot justify post-hoc method selection. Full per-sample decisions and trace hashes
are in `artifacts/sensitivity/` and `artifacts/results/SUFFICIENCY_SENSITIVITY.*`.

### Frozen OpenBookQA stress test

After the v5 policy and perturbation generators were frozen, Llama was run on the
official 500-example OpenBookQA test split, a deterministic option permutation, and a
deterministic distractor-context variant. Inference inputs were physically separated
from answer keys before 3,720 real model requests were issued.

| Variant | QPCF | Plain RAG | Same-budget majority | QPCF - plain |
|---|---:|---:|---:|---:|
| Original | 83.40 | 85.40 | 84.00 | -2.00 |
| Option-permuted | 83.20 | 86.60 | 81.80 | -3.40 |
| Distractor context | 84.00 | 87.00 | 83.60 | -3.00 |

This is a retained negative result. The v5.1 IDK sufficiency rule is structurally
inactive on these files because they contain no explicit abstention option, and replay
confirms that every OpenBookQA prediction is unchanged. OpenBookQA's `fact1` often supplies only one
premise and leaves a routine common-knowledge bridge unstated. QPCF is stable under the
two perturbations, but its conflict correction over-intervenes relative to plain RAG in
this partial-evidence setting. The result rules out a universal non-conflict-improvement
claim; it does not invalidate the paired Golden/counterfactual conflict result. See
`docs/HOLDOUT_AUDIT.md` and `artifacts/results/HOLDOUT_RESULTS_llama.md`.

The 14 primary entries are model-dataset pairs: Llama runs all six datasets, while
Qwen and Mistral each run the four non-Golden datasets. Every pair has a full
per-sample JSONL file under `artifacts/projections/`.

For a consistent baseline and ablation scope, all aggregate comparisons use the
12 non-Golden model-dataset pairs. `plain_context` is the unified standard-RAG
baseline; same-observation majority and provenance-grouped voting are budget-matched
controls. Full QPCF exceeds majority on 10 pairs and trails it on 2, with a
sample-weighted effect of +1.017 percentage points. The effect is not positive for
every model, so the release does not claim universal dominance over majority voting.

The stricter `shared_backbone_capacity` sensitivity groups every LLM call under one
capacity edge. It is 0.68 points worse in sample-weighted ACC under the same four-dataset
scope, but its direction
is inconsistent and it lowers exact-checkpoint Mistral FaithEval from 84.3 to 82.6.
It is reported as a dependence sensitivity analysis, not selected after inspecting
pair-level outcomes.

The Llama Golden and counterfactual files are aligned pairs with identical questions
and choices but disjoint correct options. Requiring both members of each pair to be
correct gives the following secondary option-level analysis:

| Pair | QPCF joint | Plain RAG joint | Majority joint | Provenance-grouped vote joint |
|---|---:|---:|---:|---:|
| MuSiQue | 83.07 | 70.94 | 80.81 | 79.97 |
| SQuAD | 87.56 | 73.88 | 86.55 | 85.02 |

QPCF significantly exceeds provenance-grouped voting on both pairs after Holm
correction. Its advantage over ordinary same-observation majority is significant on
MuSiQue but not on SQuAD. Full per-pair effects and intervals are in
`artifacts/results/PAIRED_COUNTERFACTUAL.md`.

## Layout

- `code/src/ugcp_v5/`: semantic grounding, graph evidence fusion, replay, gates, and statistics.
- `code/source_generation/`: exact source-generation code used for the formal traces.
- `code/tests/`: gold-invariance, flow, permutation, and evaluator-integrity tests.
- `data/`: six evaluated data files.
- `artifacts/source/`: real vLLM plain and graph traces with server metadata.
- `artifacts/semantic/`: frozen semantic-grounding outputs.
- `artifacts/projections/`: per-sample decisions, baselines, and ablations; the path name is retained for compatibility.
- `artifacts/results/`: aggregate metrics and paired internal statistics.
- `artifacts/results/PAIRED_COUNTERFACTUAL.*`: paired Golden/counterfactual option analysis.
- `artifacts/results/PROVENANCE_REPLICATION.*`: within-source observation-replication check.
- `artifacts/results/HOLDOUT_RESULTS_llama.*`: frozen OpenBookQA accuracy, robustness, interventions, and trace audit.
- `artifacts/results/PAPER_REPORT.*`: unified paper-facing report across all evidence tiers.
- `artifacts/results/SUFFICIENCY_SENSITIVITY.*`: fresh-run threshold sensitivity tables and trace audit.
- `artifacts/sensitivity/`: label-isolated inputs, raw requests and responses, semantic traces, decisions, and logs for the threshold study.
- `artifacts/results/SOURCE_CHANNEL_ABLATION.*`: complete one-source-at-a-time ablation table and paired tests.
- `artifacts/source_ablation/`: fresh label-isolated source traces, semantic records, decisions, and logs for source-channel ablation.
- `artifacts/results/MAIN_COMPARISON.*`: paper-style comparison with the requested baselines.
- `code/scripts/run_baselines.sh`: 12-worker zero-shot rerun for w/o RAG, RAG, and KRE.
- `code/scripts/run_sufficiency_sensitivity.sh`: fresh per-sample Llama run for graph and IDK thresholds 2, 3, and 4.
- `code/scripts/run_source_channel_ablation.sh`: fresh per-sample Llama run for all six source-family removals.
- `docs/BASELINES.md`: baseline definitions and provenance boundaries.
- `artifacts/results/RELEASE_AUDIT.*`: machine-readable completeness, provenance, and claim audit.
- `artifacts/development/`: explicitly post-hoc diagnostic probes, excluded from the v5.1 method.
- `artifacts/manifests/`: release integrity hashes.

## Replay

```bash
python -m pip install -r code/requirements.txt
PYTHON=python bash code/scripts/verify_release.sh
```

The verification entry point runs the unit tests and then uses the included frozen
source and semantic traces to regenerate all 14
per-sample decision files, checks every model gate, rebuilds the 10,000-repetition paired
statistics report, verifies source-internal observation-replication invariance, audits the
release, rebuilds the integrity manifest, and verifies all manifest hashes.
Source generation requires an OpenAI-compatible vLLM server and is documented in
`docs/REPRODUCIBILITY.md`.

## Evidence Boundary

These results are reproducible adaptive development-trace replays, not a previously
untouched benchmark holdout. Paired McNemar and bootstrap tests apply only to UGCP's
same-sample internal comparators and ablations. Without FaithfulRAG or TruthfulRAG
per-sample predictions, no paired significance claim is made against either paper.

The OpenBookQA run is a pre-inference-frozen stress test of the v5 policy. It is not a
fresh confirmatory holdout for v5.1; it serves as a regression check because the new
IDK rule cannot activate when no abstention option is present.
