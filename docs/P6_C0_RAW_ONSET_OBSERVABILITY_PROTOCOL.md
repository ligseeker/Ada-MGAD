# P6-C0 Train raw-input onset observability audit

**Evidence grade: post hoc descriptive Fit/Validation development.** This
audit asks whether the frozen normalized Metric, Log and Trace inputs show a
simple change in the first 60 seconds of an isolated GAIA event. It does not
fit a detector, search features, inspect Test, or establish causal visibility.
The C0 decoder, score, threshold, GT and Test verdict remain unchanged.

## Scope and fixed statistic

Use only the original 70% Train-side preprocessed arrays and the checksum-bound
C0F Fit/Validation observed matching. Recover each complete event exactly once
from that matching. `clean_context` means no **other complete in-block** GT
event interval intersects `[onset-60s,onset+60s]`; exclude cases lacking a
full 300-second in-block prehistory or a legal 60-second post-onset grid slot.
This in-block definition can miss a purged cross-boundary event and is not a
proof of an uncontaminated physical injection. Report its denominator.

At each event onset `t`, use the last ten 30-second bins ending no later than
`t` as a 300-second baseline. For each modality separately, take the median
feature tensor over those bins. In every 30-second bin with prediction time
`u` satisfying `0 < u-t <= 60s`, compute the elementwise absolute difference
from that baseline over all services and features. Two fixed statistics are
recorded: mean absolute difference and maximum absolute difference; take the
largest statistic across eligible post-onset bins. No root label is used to
select a service, and no feature is selected after seeing outcomes.

For each split, construct a reference pool by setting pseudo-onset `t` to
`prediction_time - 15s` for each existing legal score-grid time, with full
in-block 300-second prehistory and 60-second
follow-up, with no complete in-block GT interval intersecting
`[t-300s,t+60s]`. Require at least 512 eligible times and select exactly 512 pseudo-onsets with one
fixed PCG64 seed `20260930`; the selection uses no model score or modality
value. Calculate the same six statistics. Report count and quantiles for
clean-context `login_failure`, clean-context `memory_anomalies`, all other
clean event faults together, and pseudo-onsets. For each event group/statistic,
report descriptive Mann-Whitney AUC against the pseudo-onsets. No p-values,
thresholds, best-modality selection or independent performance claim.

The event-group contrast is limited by the GAIA fault/service/duration
confounding, different group sizes, the simple global summary statistic and
Train-fitted normalization. An AUC near 0.5 does not prove raw telemetry is
uninformative; an AUC above 0.5 does not prove a detector can identify the
event within the 60-second causal tolerance. The purpose is to decide whether
a targeted retraining hypothesis has a concrete input signal worth testing.

## Identity and execution

Pin the four Train array SHA-256 values:

| array | SHA-256 |
|---|---|
| `timestamps.npy` | `ed06402501a887099df16b0353ef4825beda2f24a74d6dae6c38b7d34de91735` |
| `metric.npy` | `8fb92322f56007f59007b102f6eb07b0cad6496b44b4292f63ced0d797956ee0` |
| `log.npy` | `02af8ad63423cfaebadc336f122c984a0a6b7c718db4674c62a6ddd40f9ff9dc` |
| `trace.npy` | `fd49c42223e5dd99420bb5e13432b72c801e6c86e5562620e938599f74d628f2` |

Also bind the C0F Fit/Validation score and observed-matching hashes from
`scripts/p6/audit_c0r2_trainval_dev.py`. The raw arrays and C0F directory
are read-only. Commit protocol, implementation and tests before execution.
Write once to `experiments/p6/c0_raw_onset_observability/raw-onset-v1/` in the
isolated `experiment/p6-c0-raw-observability` worktree, recording execution
commit, source/input hashes and Python/numeric-library versions. Stop on
misaligned timestamps, non-finite values, a GT identity mismatch, or an
existing output directory. The historical C0F Fit/Validation inference
source remains `UNVERIFIED`.
