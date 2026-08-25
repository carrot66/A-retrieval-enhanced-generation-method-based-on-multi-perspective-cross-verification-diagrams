# Formal Properties

The implementation provides four auditable properties.

1. **Label isolation:** source generation receives inference records without
   `answer` or `answerKey` fields. Gold labels are joined only for evaluation.
2. **Source capacity:** a source node has capacity one in the evidence graph, so
   several fields emitted by one model call cannot create several independent
   support units for the same option.
3. **Candidate-blind verification:** verifier prompts do not expose the candidate
   selected by another branch. A verification result must recover its option from
   the supplied context or evidence path.
4. **Fresh trace audit:** every formal result records request counts, source hashes,
   raw responses, worker count, failure count, and label-isolation metadata.

The graph decision is deterministic after the model and semantic observations have
been recorded. It is not a factual knowledge graph and does not claim statistical
independence between LLM branches that share a backbone.
