# Method

## Overview

QPCF (Query-aware Provenance-Constrained Fusion) is a training-free method for
multiple-choice RAG when retrieved evidence may conflict with an LLM's parametric
knowledge. `UGCP-v5.1-QPCF` is the release identifier used by the code and artifacts.

The central idea is straightforward: ask the LLM and two frozen semantic models for
several complementary views of the same RAG instance, ground those views in the
retrieved context, and combine support by provenance-distinct source family. The method
changes a context-aware fallback answer only when multiple source families identify one
uniquely supported option.

QPCF contains four modules:

1. Multi-view RAG evidence generation.
2. Query-aware evidence grounding.
3. Provenance-aware graph evidence fusion.
4. Conservative conflict-aware answer selection.

## Problem Setting

For a question `q`, retrieved context `c`, and candidate answers `Y`, the model must
select one answer from `Y`. The context is treated as the evidence available to the RAG
system, including cases where it describes a hypothetical world that conflicts with the
LLM's prior knowledge. QPCF is an evidence-fusion method, not a general-purpose
knowledge-graph construction method.

## Concept Boundary: Assertions and Observations

QPCF uses **assertion** in the strict propositional sense. An assertion is a
declarative proposition whose support, contradiction, or uncertainty can be assessed
relative to the retrieved context or the hypothetical world described by that context.
A text span, graph node, or model score does not become an assertion merely because the
method consumes it.

- An **atomic assertion** expresses one independently checkable fact. It can often be
  normalized as a subject-relation-object triple, but a triple representation is not
  required and unary properties also qualify. A sentence containing two independently
  truth-evaluable facts is not atomic.
- A **composite assertion** explicitly combines multiple atomic assertions through a
  conjunction, condition, causal relation, or another operator.
- An **evidence span** is text copied or located in the retrieved context. It supplies
  provenance; it is also an assertion only when it expresses a complete proposition.
- An **answer hypothesis** is the proposition that option `y` answers question `q`. It
  is a candidate conclusion, not a factual assertion extracted from the context.
- An **answer observation** records that a particular LLM or semantic branch selected
  option `y`. It describes source behavior and is not evidence that `y` is factually
  correct merely because a model produced it.
- A **semantic signal**, such as a similarity score, NLI score, or selected option ID,
  is an alignment signal rather than an assertion.

The current QPCF graph contains source-family, answer-observation, and answer-option
nodes. It does **not** extract or traverse factual assertion nodes and therefore should
not be described as an assertion graph. A future SAS-inspired extension could replace
sentence-level evidence units with source-grounded atomic assertions, but model votes
must not be relabeled as assertions and such an extension is not part of the current
reported method.

## 1. Multi-View RAG Evidence Generation

The first module collects complementary answer signals rather than relying on one LLM
generation:

- **Context-aware answer:** a standard zero-shot RAG call answers from the retrieved
  context.
- **Structured reasoning view:** one LLM call separately returns its initial answer,
  context-supported answer, parametric-knowledge answer, and an exact supporting span.
- **Candidate-blind verification views:** when the initial and context-supported
  answers disagree, two separately prompted verification calls recover an answer from
  the full context and from the extracted evidence span. They are not shown the answer
  selected by the earlier branch.

The verifier prompts do not reveal another branch's selected option. This reduces
confirmation bias and makes agreement between branches more meaningful. No branch sees
the gold label.

These outputs become typed evidence nodes. The graph records which answer each branch
supports and which model call produced that support.

## 2. Query-Aware Evidence Grounding

The second module checks whether candidate answers are semantically grounded in the
retrieved context using two frozen encoders:

- **Dense context matching:** `all-MiniLM-L6-v2` finds the context sentence most
  semantically similar to each answer option.
- **Question-conditioned entailment:** `nli-deberta-v3-small` scores whether a context
  sentence entails the statement `The answer to the question '<question>' is
  '<option>'.`

The first signal captures broad semantic overlap. The second explicitly includes the
question, so a topically similar sentence is not automatically treated as answer
evidence. Both encoders run in inference mode with frozen parameters; there is no
fine-tuning, optimizer, gradient update, or label fitting.

## 3. Provenance-Aware Graph Evidence Fusion

The third module combines the evidence signals while preventing correlated fields from
one call from being counted as separate votes. Its directed graph has three layers:

```text
source -> evidence source -> observation -> answer option
```

Each evidence source receives capacity one. Therefore, multiple fields generated by the
same LLM call contribute at most one unit of support. Plain RAG, the structured LLM call,
the two candidate-blind verifiers, dense matching, and question-conditioned entailment
are treated as distinct operational source families.

"Distinct source family" is a provenance constraint, not a claim of statistical
independence. The LLM branches share one backbone and can remain correlated. The graph
guarantees only that multiple fields from one call cannot inflate support.

For every answer option `y`, QPCF computes the maximum source-to-option flow `F(y)`.
This value is the number of source-family-disjoint support paths that can reach `y`
without double-counting one source family. Conflict-aware edge masking removes
prior-biased routes only when a context-prior disagreement is present and the
question-conditioned NLI branch independently selects the context-supported option.
An unconfirmed disagreement leaves the prior-side routes intact.

An option is eligible for selection only when it is the unique maximum-flow option and
has support from at least two source families. Ties and weak support are treated as
insufficient evidence. This is why the graph is more than ordinary majority voting:
majority counts every observation equally, whereas QPCF groups correlated observations
by provenance and changes graph connectivity under a detected conflict. The release
still reports a same-observation majority baseline to measure this difference directly.

## 4. Conservative Conflict-Aware Answer Selection

The final module is a two-route decision policy:

1. **Standard conflict route:** when no explicit abstention option is present, begin
   with the context-grounded fallback and change it only when graph fusion is active
   and one answer has unique multi-source support.
2. **IDK sufficiency route:** when an explicit `I don't know` option is present, treat
   the plain RAG answer as the candidate claim. Emit a concrete answer only when at
   least three provenance-distinct source families support that same candidate;
   otherwise output `I don't know`.

The IDK quorum is fixed globally, does not depend on model or dataset identity, does
not inspect gold labels, and adds no model call. This design directly guards against
over-answering while preserving the ordinary conflict route on datasets without an
explicit abstention option.

## End-to-End Inference

For each sample, QPCF performs the following steps:

1. Generate the context-aware answer and structured reasoning view.
2. Trigger candidate-blind verification only when the observed conflict requires it.
3. Compute dense-matching and question-conditioned entailment signals.
4. Build the provenance-aware evidence graph and compute support flow for each option.
5. Apply either the standard graph-gated route or the IDK sufficiency route and output
   the supported option, fallback answer, or explicit abstention.

All routing decisions use only the question, retrieved context, candidate answers, and
model outputs. Gold answers are joined only after the final option ID is fixed.

## Ablations and Baselines

The formal outputs include the following controlled variants, all using the same frozen
LLM observations:

- Remove question-conditioned entailment.
- Remove dense semantic matching.
- Remove both semantic grounding signals.
- Remove source-family capacity while retaining all other graph operations.
- Cap all calls from the shared LLM backbone as one conservative source group.
- Remove both source-family capacity and conflict-aware graph topology.
- Remove conflict-aware edge masking.
- Remove the conservative answer-update policy.
- Remove the IDK sufficiency protection and recover the frozen v5 decision rule.
- Use only the context-grounded fallback before graph updating.
- Use only the standard context-aware RAG answer.
- Use same-observation majority voting.
- Use provenance-grouped voting without graph masking or conservative updating.

Because these variants reuse identical model generations, paired comparisons isolate
the decision mechanism instead of confounding it with different LLM calls.

## Reader-Friendly Terminology

| Code or earlier term | Paper-facing term | Meaning |
|---|---|---|
| assertion graph | do not use | Current QPCF has no factual assertion nodes |
| typed observation graph | multi-view evidence graph | Graph containing answer signals from LLM and semantic branches |
| semantic bridge | query-aware evidence grounding | Dense matching plus question-conditioned NLI |
| provenance capacity | source-level deduplication | One model call cannot create several separate votes |
| directional cut | conflict-aware edge masking | Remove prior-biased routes only after query-conditioned confirmation |
| max-flow certificate | multi-source support score | Number of source-family-disjoint paths reaching an option |
| scope certificate | context-grounded fallback | Conservative answer selected before graph-based updating |
| risk projection | confidence-gated answer update | Change the fallback only under unique multi-source support |
| candidate-blind verifier | candidate-blind verifier | Re-evaluate evidence without seeing the earlier branch's answer |
| IDK guard | IDK sufficiency protection | Require three source families before replacing explicit uncertainty with a concrete answer |
