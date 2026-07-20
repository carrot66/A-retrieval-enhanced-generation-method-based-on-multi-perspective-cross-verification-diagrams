# QPCF Formal Properties and Boundaries

## Setup

For an answer option (y), let (F(y)) be the maximum flow from the super-source to
the option node. Every operational source family (s) is connected through one edge
with unit capacity. Observations produced by that family connect the family node to the
answer options they select.

These properties concern answer observations, not factual assertions. They do not imply
that an LLM observation is true or that two prompt branches are statistically
independent.

## 1. Within-Family Replication Invariance

If an observation supporting option (y) is duplicated any number of times under the
same source family, every support score and the selected flow winner remain unchanged.

All duplicated paths share the same unit-capacity source-to-family edge. Once one path
from that family reaches (y), additional identical paths cannot increase the flow
through the bottleneck. The release checks this property after adding eight replicas to
every active observation in all 17,503 development samples; no score or winner changes.

This property applies only to repeated answer observations within an already identified
family. It does not claim invariance to duplicated retrieval documents, paraphrases, or
new model calls whose provenance has not been grouped.

## 2. Bounded Influence of One Added Family

With graph masking held fixed, adding one new unit-capacity source family changes
(F(y)) by at most one for every option (y). The new family introduces only one new
unit of source-side capacity.

This bound does not apply when the added information also changes conflict detection or
activates query-confirmed edge masking, because that is a topology intervention rather
than a fixed-graph addition.

## 3. Option-Permutation Equivariance

If the candidate-answer order is permuted and every parsed LLM identifier, semantic
identifier, and answer label is transformed by the same permutation, QPCF's output is
transformed by that permutation. No option position receives a hard-coded preference.

The release tests this property on the decision function, and the sealed holdout
protocol includes a full option-permuted OpenBookQA split for empirical confirmation.

## 4. Gold Independence

The option decision is invariant to the gold answer field. During replay, the answer is
removed before `select_variants` is called. Gold is joined only after every variant has
fixed its option ID. A unit test changes the gold answer while keeping all observable
inputs fixed and requires identical decisions and diagnostics.

## 5. Conservative Update

QPCF retains the context-aware fallback unless all update conditions hold: the graph
route is active, no explicit abstention option blocks the update, the fallback has not
already changed through the context-grounded rule, and one option has unique support
from at least two source families. A tie, a one-family winner, or inactive routing
cannot force a graph update.

## Static-Topology Equivalence

For the current shallow graph with fixed edge masking, (F(y)) equals the number of
source families that retain at least one path to (y). Maximum flow therefore implements
provenance-constrained counting in this topology; it is not evidence of general graph
traversal or message passing.

QPCF differs from a static provenance-grouped vote through query-confirmed edge masking
and conservative answer selection. The release reports `provenance_grouped_vote` as a
same-observation comparator and does not attribute the full system gain to max-flow
alone.

## Executable Checks

- `code/tests/test_projection.py`: capacity, replication, bounded influence,
  query-confirmed masking, ties, option permutation, gold invariance, and fallback.
- `artifacts/results/PROVENANCE_REPLICATION.json`: all-sample replication check.
- `artifacts/results/PAIRED_COUNTERFACTUAL.json`: paired Golden/counterfactual
  evaluation after decisions are fixed.
