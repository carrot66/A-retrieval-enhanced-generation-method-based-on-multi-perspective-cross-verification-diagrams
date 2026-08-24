# Limitations

1. Method design inspected benchmark traces. The formal replay proves reproducibility,
   not performance on a previously untouched benchmark holdout.
2. Published FaithfulRAG and TruthfulRAG values are point-estimate gates. Their
   per-sample predictions are unavailable here, so paired significance is not claimed.
3. The main Mistral rows use the exact `mistralai/Mistral-7B-Instruct-v0.3`
   checkpoint named by TruthfulRAG and physically label-isolated inference inputs.
   Ten plain calls returned empty completion text and were retained without selective
   retry. Published targets remain point estimates because per-sample baseline
   predictions are unavailable.
4. UGCP is evaluated on multiple-choice tasks. Claims should not be generalized to
   open-ended RAG without new experiments.
5. Provenance-aware graph fusion does not beat same-observation majority on every pair
   under the unified four-dataset scope: it wins 10 of 12 and loses 2. Qwen's
   sample-weighted majority score is 0.11 points higher.
6. The source-family-capacity ablation has a +0.079 point aggregate effect and is not
   Holm-significant. The strongest evidence supports the full evidence-fusion system,
   not a claim that max-flow or source capacity alone explains all gains.
7. The current graph contains source, answer-observation, and answer-option nodes. It
   does not contain factual assertion nodes, so the method does not yet demonstrate
   atomic-assertion reasoning or general knowledge-graph inference.
8. The query-confirmed masking policy and the v5.1 IDK sufficiency rule were selected
   after inspecting development traces. OpenBookQA is not a fresh holdout for v5.1;
   because it has no explicit IDK option, it only verifies that the new rule leaves
   non-IDK predictions unchanged.
9. Provenance families are operational groups, not statistically independent evidence
   sources. Multiple LLM branches share one backbone and may have correlated errors;
   source capacity prevents within-call duplication but does not remove this dependence.
   The `shared_backbone_capacity` sensitivity variant groups all LLM calls under one
    capacity edge. It has mixed pair-level effects and lowers exact-checkpoint Mistral
   FaithEval from 84.3 to 82.6. Its aggregate gain and mixed pair-level direction are
   reported as sensitivity evidence; it is not selected after observing outcomes.
10. Paired Golden/counterfactual joint accuracy is available only for Llama under the
    formal protocol. Its advantage over same-observation majority is Holm-significant
    on MuSiQue but not on SQuAD.
11. On the frozen OpenBookQA partial-evidence stress test, QPCF trails matched plain RAG
    by 2.0 points on the original split, 3.4 points after option permutation, and 3.0
    points with a distractor context. The method rescues 10 but harms 20 original-split
    decisions. QPCF therefore does not support a universal non-conflict-improvement
    claim.
12. OpenBookQA was untouched for the reported frozen v5 run, but its outcomes have now
    been inspected. It cannot be reused as a confirmatory holdout for a modified method.
    Post-hoc proof-completeness, contrastive-adjudication, and NLI-abstention probes are
    retained under `artifacts/development/` and are not part of QPCF-v5.1.
13. The fresh threshold sensitivity run gives higher RealtimeQA ACC at an IDK minimum
    of 4 than at the frozen value of 3. This is a post-hoc test-set observation, not a
    valid basis for changing the method. The release therefore retains 3 and reports
    all thresholds without selecting the best test result.
