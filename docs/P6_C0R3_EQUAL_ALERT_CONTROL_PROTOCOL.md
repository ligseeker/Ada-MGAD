# P6-C0R3: equal-alert-budget decoder control and onset-score feasibility

**Evidence grade: Fit/Validation development only, post hoc.** The positive-rise
decoder and its Fit/Validation outcomes are already known. This audit tests a
specific competing explanation, not an independent generalization claim. It
must never read Test, change the frozen C0 checkpoint/threshold, retrain, or
alter the raw GT and exact 60-second causal one-to-one evaluator.

## Question 1: does score rise choose useful alert locations?

Replay the C0F Fit and Validation fixed score files, original episode decoder,
and C0R2 positive-logit-rise decoder exactly. Fail closed unless the original
baseline reproduces every archived `(case_id,t_hat)` match and C0R2 candidate
reproduces archived TP/FP/FN and emitted-alert counts.

For each split, partition continuous above-threshold bins into positive runs.
For every run length `L`, count the C0R2 extra alerts emitted at noninitial
positions among **all** runs of length `L`. Each placebo replicate samples that
same count uniformly without replacement from every eligible noninitial bin
of all runs of length `L`, with no GT, fault type or root information. The run
starts stay fixed. This preserves total alerts and the alert budget by run
length while randomizing both which run and which within-run position receives
an extra alert. Matching, precision, recall, F1 and raw GT denominators are
identical. Use 64 fixed NumPy PCG64 seeds `20260930..20260993`; report all
replicates and their min/median/max and 5th/95th percentiles, with no seed
selection. No statistical p-value is claimed.

Report the same quantities within pre-defined onset groups: `isolated_onset`
means no other GT onset in the closed +/-60-second neighborhood;
`clean_context` means no other GT event interval intersects
`[onset-60s,onset+60s]`. Both are evaluation-only strata, not decoder inputs.
In particular, compare C0R2 with the placebo distribution for recall on
clean events and for total event F1 at the exact same alert count. If C0R2
does not beat the placebo 95th percentile in both Fit and Validation on total
F1 **and** clean-context recall, treat the previous recall gain as
alert-budget sensitivity, not evidence of onset-specific timing. Passing this
development gate would only motivate an independent future validation.

## Question 2: is the existing score responsive at long/memory onsets?

Using only the same sealed Fit/Validation score series and observed matching,
report by fault type, duration and clean-context status: case count, baseline
matched count, number with at least one legal score in `[onset,onset+60s]`,
number with score above the frozen threshold, and distribution of the maximum
score in that causal interval. Preserve all cases in denominators. This
describes the **current model score**; a low value cannot prove raw telemetry
is unobservable or that retraining cannot help. A future model-training design
requires a separate raw-signal audit and a frozen single-variable intervention.

## Sources, outputs and stop rules

Bind the existing C0F Fit/Validation score and observed matching SHA-256 values
from C0R2 and the prior C0R2 result SHA-256
`773f05832dcefb568cc867f3eb716c3db2be234cf57d539ede7ca6a59b5dc9f0`.
The combined C0F `event_failure_ledger.csv` also contains Test rows and is
forbidden for this audit. Commit this protocol, implementation and
tests before execution. Write once to
`experiments/p6/c0r3_equal_alert_control/c0r3-v1/`; preserve the previous
run and original worktree. Record execution commit, source/input hashes,
package versions, exact baseline/C0R2 replay checks and all 64 control
outcomes. Any drift, missing denominator, broken alert-budget equality or
non-finite score is a STOP. The historical Fit/Validation score inference
source remains `UNVERIFIED` despite this new audit's source lock.
