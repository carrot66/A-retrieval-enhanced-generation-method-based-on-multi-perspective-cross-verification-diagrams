# QPCF Novelty Audit

Audit date: 2026-07-11.

## Scope and Search Procedure

The audit searched arXiv, OpenAlex, and available paper source files for work published
or posted from 2023 through July 2026. Queries covered RAG knowledge conflict,
provenance-aware fusion, evidence graphs, argumentation graphs, atomic claims, and
network flow. Venue status below follows the authors' arXiv metadata and is not treated
as independent confirmation.

## Closest Work

| Work | Main technical object | Relation to QPCF |
|---|---|---|
| [FaithfulRAG](https://arxiv.org/abs/2506.08938) (ACL 2025) | Fact-level context/parametric conflict modeling and self-thinking | Same conflict setting; QPCF does not perform its fact comparison process |
| [Astute RAG](https://arxiv.org/abs/2410.07176) (ACL 2025) | Iterative source-aware consolidation of internal and external knowledge | Makes source-aware consolidation alone an insufficient novelty claim |
| [Micro-Act](https://arxiv.org/abs/2506.05278) (ACL 2025) | Hierarchical fine-grained actionable comparisons | Makes fine-grained multi-step conflict reasoning alone insufficient |
| [TruthfulRAG](https://arxiv.org/abs/2511.10375) (AAAI 2026) | Retrieved-context KG, query-based paths, and entropy filtering | Establishes graph-based factual conflict resolution before QPCF |
| [ArgRAG](https://arxiv.org/abs/2508.20131) (arXiv 2025) | Quantitative bipolar argumentation graph over claims and evidence | Establishes deterministic support/attack graph reasoning for noisy RAG |
| [Atomic Information Flow](https://arxiv.org/abs/2602.04912) (arXiv 2026) | Atomic tool information in a network-flow/min-cut model | Makes “atomic units plus network flow” a directly overlapping formulation |
| [ArbGraph](https://arxiv.org/abs/2604.18362) (arXiv 2026) | Atomic claims, support/contradiction edges, and iterative credibility propagation | Directly covers the full SAS-style atomic-claim evidence-graph direction |
| [Seeing through the Conflict](https://arxiv.org/abs/2601.06842) (arXiv 2026) | Semantic match, factual consistency, and self-answerability signals | Overlaps QPCF's separation of retrieval match and question-conditioned support |
| [ERA](https://arxiv.org/abs/2604.20854) (under review in metadata) | Evidence distributions and Dempster-Shafer conflict quantification | Establishes explicit multi-source belief fusion for RAG reliability |
| [ConflictRAG](https://arxiv.org/abs/2605.17301) (submitted in metadata) | Learned conflict detection and source-credibility resolution | Broader conflict taxonomy, but uses training and tuned thresholds unlike QPCF |

## What Is Not Novel Enough

The following descriptions should not be used as QPCF's primary contribution:

1. “The first graph method for RAG knowledge conflict.” TruthfulRAG, ArgRAG, and
   ArbGraph contradict this claim.
2. “The first atomic-assertion graph for RAG.” ArbGraph already decomposes documents
   into independently verifiable claims with support and contradiction edges.
3. “The first network-flow model for RAG.” Atomic Information Flow already formulates
   RAG tool attribution and compression with atomic network flow and min-cut.
4. “The first source-aware fusion method.” Astute RAG and later reliability work use
   source-aware consolidation or belief fusion.
5. “Graph reasoning is the main cause of the gain.” In the current development replay,
   removing source-family capacity changes sample-weighted ACC by only +0.034 points
   in favor of full QPCF and is not Holm-significant.

## Defensible Difference

The most defensible current positioning is narrower:

> QPCF is a training-free, label-free decision method for multiple-choice
> context-parametric RAG conflicts. It represents LLM and frozen-encoder outputs as
> answer observations rather than factual assertions, prevents multiple fields from one
> call from being counted repeatedly, masks prior-side routes only after
> question-conditioned confirmation of the context candidate, and otherwise retains a
> context-aware fallback.

This differs from factual KG construction, atomic-claim arbitration, argumentation
semantics, learned conflict detectors, and trained flow compressors. The distinction is
real, but it is a narrow systems-and-decision contribution rather than a new general
graph-reasoning paradigm.

## SAS Decision

Importing SAS wholesale is not recommended. The combination “atomic assertions,
query-aware filtering, support/contradiction graph, graph arbitration” would move QPCF
toward ArbGraph, while combining those assertions with max-flow would also move it
toward Atomic Information Flow.

The only technically clean borrowing is the representation principle: if a future
grounding experiment replaces a sentence with a source-linked atomic proposition, that
proposition must be a single independently verifiable fact. This would replace the
sentence-level grounding unit inside Module 2; it must not be added as a fifth module,
and model votes must remain answer observations. Such a change is not part of the
current reported results and requires a sealed holdout evaluation.

## Current Publication Assessment

Current evidence is stronger than the development-only package but still does not
justify an unconditional “AAAI-ready” claim:

- All 14 configured development pairs exceed the published point-estimate gates.
- Under the unified four-dataset baseline scope, QPCF beats same-observation majority
  on 10 of 12 pairs, but Qwen is 0.11 points lower in sample-weighted ACC.
- Paired Llama joint option accuracy is 83.07 on MuSiQue and 87.56 on SQuAD. QPCF
  significantly exceeds provenance-grouped voting on both; its gain over ordinary
  majority is not Holm-significant on SQuAD.
- Source-internal observation replication leaves all 17,503 support vectors and winners
  unchanged, as required by the provenance-capacity design.
- The frozen OpenBookQA model run is complete: 1,500 sample variants, 3,720 real Llama
  requests, zero failures, complete request/response traces, and physical label
  isolation. QPCF scores 83.4/83.2/84.0 on original/permuted/distractor variants.
- The same holdout falsifies universal non-conflict improvement: matched plain RAG
  scores 85.4/86.6/87.0. QPCF's original-split interventions rescue 10 and harm 20.
- The shallow flow graph is close to provenance-grouped counting; its strongest
  empirical gains come from semantic grounding and conservative conflict-aware
  selection rather than source capacity alone.
- Grouping every LLM call under one shared-backbone capacity edge is 0.68 points lower
  under the unified four-dataset scope, but has mixed effects and lowers exact-checkpoint Mistral
  FaithEval from 84.3 to 82.6. This sensitivity result supports reporting cross-call
  dependence, not switching the policy after observing model or dataset outcomes.

OpenBookQA is confirmatory evidence for the frozen v5 policy and only a non-IDK
regression check for v5.1, because the new rule cannot activate without an explicit
abstention option. Three post-hoc candidates were tested and rejected rather
than folded into v5.1: a strict proof-completeness critic removed conflict gains, an
anonymous contrastive adjudicator also damaged conflict accuracy, and model-native NLI
abstention gave only small OpenBookQA gains while lowering most development rows.

## Recommended Paper Claim

Given the completed frozen holdout, the strongest accurate claim is:

> On frozen development traces from the FaithEval, RealtimeQA, MuSiQue, and SQuAD
> conflict suites, QPCF provides zero-training, gold-independent, provenance-constrained
> answer fusion that exceeds the configured published point estimates across three
> backbones. Paired Golden/counterfactual analysis shows that its gains are not obtained
> solely by sacrificing the non-conflict member, while same-observation voting baselines
> delimit the contribution of graph-based decision logic. A separately frozen
> OpenBookQA stress test exposes a partial-evidence limitation: QPCF remains stable under
> option and context perturbations but trails matched plain RAG.
