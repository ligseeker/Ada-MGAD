# P6-C0R2 Fit/Validation positive-rise feasibility audit

**Evidence grade: development-only, post hoc.** The frozen P6-C0 Test has
already been inspected in earlier research. In this turn the agent also
viewed Fit and Validation aggregates for two decoder ideas before writing
this protocol. This document locks a reproducible replay of that development
observation; it does **not** create a pre-registered or independent
Validation result.

## Question and single candidate

Can the existing fixed C0 system score support more causal event triggers
without retraining? Keep the checkpoint score bytes, frozen threshold
`0.9998264908790588`, 30-second grid, 60-second causal matching, raw GT,
Fit/Validation split and evaluator unchanged. The only changed variable is
episode emission:

- Baseline: emit at the first positive bin of a contiguous positive run.
- Candidate: emit at that bin and at a later positive bin **only if its logit
  strictly exceeds the previous positive bin's logit**. A rise is an observed
  score change, not a root/fault-label condition. Each emission is a separate
  one-bin alert.

There is no new threshold, offset, trained parameter, checkpoint choice,
feature schema, Test prediction or Test evaluation. The candidate is the
minimal rising-score decoder, not the less selective each-positive-bin idea
examined during development. No variant is selected from the eventual v2
output.

## Inputs, evaluator and fail-closed gates

Use the completed C0F correction's immutable Fit and Validation score bytes
and observed matching. The script binds each input SHA-256, checks contiguous
chronology, the frozen threshold and score-to-binary equivalence. It extracts
the exact complete raw GT universe from the observed matching. It replays the
unchanged baseline through `match_events` and requires the same TP/FP/FN **and
the same `(case_id, detected_time)` pairs** before evaluating the candidate.

For runtime only, split both GT onsets and predicted anchors into temporal
components separated by more than 60 seconds; there can be no causal match
across such a gap. Inside each component call the unchanged exact
max-cardinality/minimum-delay matcher. A synthetic test verifies component
equivalence. Outputs are written once to
`experiments/p6/c0r2_trainval_dev/positive-logit-rise-v2/` after committing
the script, test and this protocol. Store the execution commit and source
hashes; never overwrite an existing audit directory.

## Interpretation and stop gate

Report Fit and Validation event P/R/F1, TP/FP/FN, exact gained/lost GT case
counts and fault mix. Audit the gained events for ongoing episodes,
other nearby onsets and same-bin overlap. An aggregate gain is not evidence
that the system model identified a second onset: a rising score inside one
anomaly may make a duplicate alert that the one-to-one matcher assigns to a
nearby injection. This is especially important for GAIA's 30-second grid.

If gains are dominated by overlapped `login_failure` while long/memory recall
barely changes, classify the decoder as **protocol sensitivity** and do not
retrain the detector on this evidence alone. A new detector method would
require a separate Train-only design lock, distinct-onset target and untouched
external evaluation. The present Test must not select the decoder or measure
its apparent improvement.

The reused Fit/Validation score bytes are checksum-bound, but the historical
source bytes that generated them were not archived; their inference source
identity remains `UNVERIFIED`. The v2 audit source can be sealed, but it
cannot retroactively repair that limitation.
