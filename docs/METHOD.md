# Method

## Paper-facing name

QPEF-RAG is Query-aware Provenance-guided Evidence Fusion for
Retrieval-Augmented Generation. The source package keeps the historical QPCF
and ugcp_v5 identifiers for compatibility with the released scripts.

## Pipeline

1. Multi-view generation produces a standard context answer, a structured
   context-versus-prior view, and candidate-blind verification observations.
2. Query-aware grounding evaluates dense semantic matching and
   question-conditioned entailment for each candidate.
3. Evidence-channel fusion groups observations by their producing process and
   applies a channel-level capacity constraint so one process cannot contribute
   several separate votes.
4. Conservative answer selection changes the fallback answer only when one
   candidate has unique and sufficient cross-channel support. Tasks with an
   explicit abstention option require a higher support threshold before a
   concrete answer is returned.

The graph stores source-process observations and candidate options. It is an
evidence-fusion graph, not a factual knowledge graph. Source channels are
operationally distinct and are not assumed to be statistically independent.