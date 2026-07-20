# Frozen OpenBookQA Audit

## Protocol Status

The QPCF-v5 decision policy, OpenBookQA conversion, option permutation, distractor
construction, semantic bridges, and model metadata were hashed before source inference.
The inference files contain no `answer` or `answerKey` fields. Answer keys were joined
only after all option IDs were fixed.

- Model: Meta-Llama-3.1-8B-Instruct.
- Split: official OpenBookQA `test_complete`, 500 examples.
- Robustness variants: deterministic option permutation and deterministic appended
  distractor context.
- Parallelism: 12 workers.
- Sampling: temperature 0, fixed per-sample seeds.
- Training: none.
- Source requests: 3,720.
- Failed requests: 0.
- Missing request/response payloads: 0.
- Label fields in source traces: 0.

## Results

| Variant | QPCF ACC | 95% Wilson CI | Plain RAG | Majority | Prediction invariance |
|---|---:|---:|---:|---:|---:|
| Original | 83.40 | [79.89, 86.40] | 85.40 | 84.00 | - |
| Option-permuted | 83.20 | [79.67, 86.22] | 86.60 | 81.80 | 85.80 |
| Distractor context | 84.00 | [80.53, 86.95] | 87.00 | 83.60 | 92.00 |

QPCF is robust in aggregate accuracy under both perturbations, but it does not beat the
matched plain-context baseline. The paired full-minus-plain effect is -2.0 points on the
original split, -3.4 on the option permutation, and -3.0 with distractors. The latter
two are Holm-significant within this three-comparison family.

## Intervention Audit

| Variant | Changed | Rescued | Harmed | Net rescues |
|---|---:|---:|---:|---:|
| Original | 40 | 10 | 20 | -10 |
| Option-permuted | 43 | 10 | 27 | -17 |
| Distractor context | 35 | 5 | 20 | -15 |

OpenBookQA's `fact1` frequently states one relevant premise but omits a routine bridge
needed to select an answer. QPCF's context-only challenger can mistake this relevance
for a complete answer path. This explains why conservative conflict correction can be
useful on explicit counterfactual contexts yet over-intervene on partial evidence.

## Post-Hoc Boundary

After observing these outcomes, three label-free candidate modifications were evaluated
as development probes:

1. A strict explicit-premise critic rejected essentially every conflict update and
   destroyed the main conflict gains.
2. An anonymous contrastive LLM adjudicator improved some OpenBookQA changed cases but
   substantially damaged FaithEval and both Negative suites.
3. NLI posterior abstention produced only +0.2 to +0.6 points on OpenBookQA while
   reducing most development configurations.

None is part of QPCF-v5.1. Their complete per-sample requests and responses are retained
under `artifacts/development/`. OpenBookQA remains a valid frozen test of unchanged v5.
For v5.1 it is only a regression check: no sample has an explicit IDK option, so the new
IDK sufficiency rule is inactive and all three prediction files remain unchanged.
