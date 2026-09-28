# P6 next experiment plan: detector-aligned RCA and full E2E

## Material Passport

- Origin Skill: `academic-research-suite/experiment-agent`
- Origin Mode: `plan`
- Origin Date: 2026-09-23
- Verification Status: G1 static inputs verified; G2 design locked; future experiment results remain `UNVERIFIED`
- Version Label: `p6_next_plan_v2`
- Status: **research sequence with a linked G2 design lock; no C1 experiment result**

## 1. Research question and priority

The immediate scientific question is whether aligning RCA training with the
detector's anchor domain improves localization on detected events when the
P6-C0 Stage-1 model and its `BORDERLINE` coverage are held fixed. I propose
**C1 followed by C2** as the next main study because it directly tests the
two-stage interface and yields a full diagnosis accounting. **C0R2** remains a
separate detector-improvement study if the objective is to address low
long-event, memory and multi-onset recall. This priority is a proposed
research choice, not a conclusion selected by C0F's post hoc loss shares.

The completed [C0F correction archive](P6_C0F_CORRECTION_ARCHIVE.md) and
[current context](GAIA_P5_CURRENT_CONTEXT.md) are the evidence baseline.
P6-C0 Test event P/R/F1 is `0.9962/0.7254/0.8395`, with 4,198 TP, 16 FP and
1,589 FN among 5,787 GT events. Long-event recall is `32/229`; memory recall
is `30/204`. C0F gives failure **locations** and exact diagnostic capacity
bounds, not an identified causal remedy. P5 GT-anchor versus detected-anchor
RCA AC@1 (`0.9125` versus `0.4657`) motivates C1 but uses a different
detector/cohort, so a new C1 contrast cannot claim to reproduce or fully
explain that numeric drop.

## 2. Sequence and stop gates

| Gate | Work and output | Stop condition |
|---|---|---|
| G0 archive | C0F code, protocol, aggregate evidence and local detailed run are archived at `eac0f44677ba1baba6b2ecac2e7b8e0e606856bd`; retain the correction limitations. | A required file/hash no longer matches the completion manifest. |
| G1 C1 feasibility | Static inventory completed in the [G1 report](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/final_report.md). It maps chronology, GT cohort upper bounds, fitting sources and open OOS/Train-case gates. | Current shared arrays cannot support strict forward-OOS; actual OOS anchor coverage and common Train RCA cohort remain unknown. Execution is NO-GO as is. |
| G2 C1 protocol freeze | [G2 design lock](P6_C1_G2_FROZEN_DESIGN.md) fixes folds, evidence grade, fit/selection/generation, cohort floors, shared scaler, arms, metrics, paired analysis, budget, run ID and firewall before results. Current Python 3.8-compatible source bindings use [v1.1](../configs/e2e/gaia_p6_c1_g2_v1_1.json); the [correction record](../experiments/p6/c1_protocol/c1-g2-py38-correction-20260924T0831Z/correction_report.md) keeps v1 historical. | A binding or frozen choice changes; version the design before any fold result. |
| G3 C1 implementation and smoke | [Fold-prefix materializer, label-free Generation dataset and synthetic orchestration smoke](P6_C1_G3_IMPLEMENTATION_STATUS.md) are implemented. Next: real raw small-scope smoke, workers=1/24 equivalence, fold detector training/inference, OOS cohort, shared-scaler RCA, label-free prediction lock and post-lock evaluator. | Identity, ordering, label isolation, Train-only fitting, window or schema checks fail. |
| G4 C1 execution | Use new run directories for necessary fold detector fitting, RCA fitting and ranking generation. Read the frozen C0 Test predictions/episodes without rerunning Stage-1 Test inference. Complete RCA rankings for all legal detected Test episodes; lock their bytes and universe before Test label join. | Required input/source/output hash changes, missing rankings, incomplete folds or budget exceedance. Preserve failure rows. |
| G5 C2 evaluation | Consume the locked predictions; publish matched-case RCA contrasts and full diagnosis results with all misses, false alarms and ranking failures. Archive hashes, commands, counts and limitations. | Prediction/scope lock or denominator checks fail. No post hoc repair of the formal result. |

G0, the **static inventory portion** of G1, and the G2 design lock are
complete. G1 did not establish actual OOS inputs or a common Train cohort;
G3 is partially implemented; G4–G5 remain pending. Read-only G2 integrity commands exist, but no C1 full
execution command exists until G3 builds and validates the required interfaces.
Full preprocessing, training, Test inference and RCA execution require an
explicit execution request. Every new run gets a unique directory.
Existing P5, C0 and C0F artifacts remain immutable.

The fixed inputs for planning are
`configs/e2e/gaia_p5_v3_preprocessing_v2.json`,
`configs/e2e/gaia_p6_c0_system_trigger.json`, the C0
`experiments/p6/system_event_trigger/{manifest.json,validation_selection.json,test_predictions.csv,test_episodes.csv,test_matching.csv}`,
the C0F correction completion manifest/decision, and the frozen shared
preprocessing roots in the current context. G2 binds exact paths and
hashes in its machine config. This list does not
authorize reuse of a mutable shared output.
The C1 intended run directory is
`experiments/p6/c1_detector_aligned/c1-prefix-oos-v1-seed42/`; C2 will receive
a separate locked directory before it runs. The current plan does not invent
a runnable C1 full-experiment entry command.

## 3. C1 controlled design

### 3.1 Intervention, controls and estimand

Keep the frozen P6-C0 Test checkpoint, threshold, grid, episode construction
and matching rule fixed. Keep the 68D Z2 feature definition, four channels,
canonical service candidates/order, RCA window, Conditional Logit form and
hyperparameters fixed. Different Train anchors entail separately fitted RCA
weights; the comparison does not change the model family.

Lock one common set of Train cases with legal GT and OOS-detected RCA contexts.
Both fitted arms use identical case IDs, supervision labels, case weights and
candidate service identities/order; feature values differ with anchor context.
Fit one StandardScaler on **Train GT-anchor candidate rows**
from that common cohort; apply the frozen scaler to all arms and evaluation
contexts. Any arm-specific scaler is a separately declared secondary analysis.

| Arm | RCA Train anchor | Evaluation anchor | Role |
|---|---|---|---|
| A | GT | GT | Oracle-anchor diagnostic on the common cohort |
| B | GT | Detected | Anchor-domain mismatch control |
| C | OOS-detected | Detected | Aligned-domain intervention |

A and B use the same trained RCA model. B and C use the same matched Test
cases for paired analysis. The primary estimand is the **paired difference in
AC@1, C minus B**, with raw correct/incorrect transition counts. Secondary
outputs are AC@3/5, MRR, root/fault macro scores and group sample sizes.
Small groups and overlapping events need declared handling and a time-aware
uncertainty method before execution. Report the full episode population and
every exclusion alongside the common-cohort comparison. A is diagnostic and
must never be presented as a deployable path.

### 3.2 Genuine out-of-sample Train anchors

Current C0 Fit predictions are in-sample. Validation participated in
checkpoint/threshold selection. Neither is a strict OOS source for RCA Train.
For each chronological fold, the detector fit interval must precede its
selection interval, which must precede the anchor-generation interval. Fold
model, threshold, calibration, schema, vocabulary, graph and scalers must fit
only on that fold's past for a claim of **full forward-OOS**.

The shared preprocessing used by C0 was fitted over the original 70% Train
and may contain future information relative to internal folds. The
[G2 design lock](P6_C1_G2_FROZEN_DESIGN.md) selects prefix-fitted detector
inputs with a frozen, label-free raw filename catalog and the fixed RCA
indicator catalog. The resulting claim is detector prefix-OOS conditional on
that transductive structural catalog; it does not claim unconstrained
full-pipeline forward-OOS. A stricter prefix-bound RCA catalog or a narrower
supervision-OOS design would require a new design version before execution.

If prefix fitting proves infeasible, record `PREFIX_SCHEMA_NO_GO`; a narrower
supervision-OOS variant needs a new design version before any run. G1's static
GT counts provide only an upper bound. G2 freezes the performance-blind
minimum actual OOS cohort, exact dates, fold count, purge and low-count
behavior before fold results. Respect both the
detector's 300-second history and the distinct
`[anchor-300s, anchor+300s)` RCA context, including injection/split crossings.
Time-match Train OOS episodes to supervision within Train and keep unmatched
episodes in the accounting.

### 3.3 Prediction and label firewall

Create the full ordered ranking for every legal detected Test episode, then
hash-lock predictions, feature inputs, model weights, source identity and case
universe. Only afterward join Test labels for the matched-case and full E2E
evaluator. Missing or invalid rankings remain failures; do not run RCA only on
known GT-matched episodes. Root/fault labels cannot enter Test features,
anchor correction, ranking or model selection. The Test set has already been
used to discover issues, so the eventual result is a **reused-Test evaluation**,
not an untouched independent confirmation.

## 4. C2 full E2E evaluation

C2 consumes G4's fixed predictions. It reports separately:

1. Stage-1 event TP/FP/FN, precision/recall/F1, detection delay and
   duration/fault/service/onset-density coverage, with raw denominators.
2. On the same eligible matched-case cohort, B/C RCA AC@1/3/5 and MRR,
   paired differences, root/fault macro and each group's `n`.
3. Full diagnosis TP/FP/FN and precision/recall/F1@k across **all** raw GT
   events and detected episodes, including detector misses, false alarms,
   boundary exclusions and ranking failures under frozen semantics.
4. Separate onset-to-detector, RCA-data-ready and final-diagnosis latency.
   The future context needed by RCA cannot be hidden inside the detector delay.
5. Exact source/input/prediction hashes, uncertainty method, Test reuse
   limitation and an evidence-graded interpretation, including NO-GO or no
   gain when supported.

Do not merge GT injections, change the matching tolerance, omit hard cases or
alter denominators to improve the result. C2 is required even if C1 shows no
benefit; a negative result still answers the interface question.

## 5. Conditional C0R2 detector branch

If improving event coverage is the next priority, freeze a **separate** C0R2
protocol. Candidate directions from the existing roadmap are (a) a
development-only selection objective for macro or minimum-group recall with
explicit false-positive and overall-performance constraints, or (b) a
system-level onset plus sustained-abnormal-state objective without node/root
supervision. Neither is selected by the C0F loss-share table or a Test sweep.

Before implementation, fix exact candidate set, target/group definitions,
overlap and small-`n` policy, Fit/Validation split, metric arithmetic,
selection/tie rules, compute budget, stopping conditions and unique run ID.
Disclose any root/fault identity labels used in development decisions. Test
cannot be used to choose the winning R2 configuration. If C0R2 changes
detected anchors or the Test cohort, the fixed-C0 C1 result does not transfer
automatically; a new matched comparison needs its own scope lock. Independent
confirmation would require a new untouched temporal period/data source or a
pre-isolated outer evaluation; reuse of the present Test remains limited.

## 6. Files to produce before execution

- `C1_FEASIBILITY_LEDGER`: [static ledger](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/feasibility_ledger.json)
  and [report](../experiments/p6/c1_feasibility/c1-g1-20260923T105441Z/final_report.md)
  are complete. Actual fold OOS anchors and the common Train cohort remain
  unknown; G2 resolved the evidence-grade choice. The G1 result was
  `EXECUTION_NO_GO_AS_IS`.
- `C1_G2_DESIGN_LOCK`: [protocol](P6_C1_G2_FROZEN_DESIGN.md) and
  [current machine config](../configs/e2e/gaia_p6_c1_g2_v1_1.json) fix the run ID,
  input hashes, folds, candidate universe, minimum cohort, shared scaler,
  arms, firewall, metrics, paired method, budget and stop conditions.
  **Design locked; implementation and smoke pending G3.**
- C1 predictions/scope lock and C2 evidence package. **Pending G4/G5.**

The [2026-09-18 roadmap](P6_RESEARCH_ROADMAP.md) preserves the original
design rationale but its statement that C0F had not run is historical. The
current status and archive identities are in the links above.
