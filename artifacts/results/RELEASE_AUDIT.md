# UGCP-v5.1-QPCF Release Audit

Status: **PASS**

## Coverage

- Formal model-dataset pairs: 14
- Formal per-sample decisions: 17503
- Retained raw model responses: 44616
- Retained empty model outputs: 10
- Frozen holdout variants: 1500
- Frozen holdout requests: 3720
- Machine checks passed: 151/151

## Claim Checks

- All 14 configured published-point-estimate gates pass.
- Training is disabled and formal workers are fixed at 12.
- The official evaluator hash matches the frozen configuration.
- No paired-significance claim is made against published systems.
- No universal non-conflict-improvement claim is made.

## Warnings

- Llama and Qwen benchmark source generation was not physically label-isolated; the exact-checkpoint Mistral and OpenBookQA runs were label-isolated before inference.
- Exact-checkpoint Mistral retained 10 empty plain completion texts without selective retry; their requests and fallback decisions remain auditable.
- Published-system comparisons are point-estimate comparisons because FaithfulRAG and TruthfulRAG per-sample predictions are unavailable.
- OpenBookQA is a retained negative result: QPCF trails matched plain RAG by 2.0 to 3.4 percentage points.
