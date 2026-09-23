# P6-C1 G1 static feasibility report

Run: `c1-g1-20260923T105441Z`

Status: **STATIC_INVENTORY_COMPLETE; C1 EXECUTION NO-GO AS IS**

Execution base HEAD: `ba0ecdb5faf61ed98a2446f553303e7b70b31668`

Audit script SHA-256: `da2779ce670141a6de8addfdb97c0918d0aa9ca6099445aae65dcbb850a166f0`

Machine ledger SHA-256: `80718e06a801f4a4cfac085eec999285a162b4e1ebf5afff082f17925a99b3da`

This is a read-only inventory of pre-Test event identities, existing GT feature
coverage and proposed chronological fold boundaries. No preprocessing,
detector/RCA training, model inference, Test-label join or performance
evaluation was run. The [machine ledger](feasibility_ledger.json) binds 14
direct input files and the audit script by size and SHA-256. It does not claim
a full-content hash of the external raw GAIA corpus; the existing raw-index
manifest's layout digest is based on filenames and byte sizes.
Four earlier local G1 probes at `104315Z`, `104558Z`, `104734Z` and `105220Z`
were superseded while adding the Fit-only alternative, service counts,
stricter Test-row filtering/input binding and a correction of the excluded
Train/Test boundary event. This `105441Z` ledger is the published reference.
All five directories remain untouched locally.

## 1. Verified timeline and population

The frozen C0/P5 timeline is a 30-second grid from 2021-07-01 10:00 UTC to
2021-07-31 16:00 UTC. C0 Fit ends 2021-07-16 13:00 UTC, Validation ends
2021-07-22 14:12 UTC, and Test ends with the full timeline. Before Test there
are 10,345 detector-domain registry rows: 7,443 complete C0 Fit events,
2,901 complete C0 Validation events and one event (`gaia-v3-10508`) crossing
the original Train/Test boundary. It is excluded from complete C0 cases.
The pre-existing GT RCA bundle has 16,124 unique cases across its original
Train/Test split, with `(case, 10 services, 68 features)` shape; its IDs,
splits and GT anchors match the GT case registry exactly.

### Preferred static candidate: three Fit-only rolling folds

These proposed endpoints divide original C0 Fit into five equal 30-second-grid
segments. Each fold uses a past detector-fit prefix, the next segment for
checkpoint/threshold selection, and a later segment only for anchor generation.
All intervals are half-open; the table uses UTC.

| Fold | Detector fit | Selection | Anchor generation | Complete GT fit/selection/generation | GT feature + GT-context eligible |
|---|---|---|---|---:|---:|
| 1 | Jul 1 10:00–Jul 4 10:36 | Jul 4 10:36–Jul 7 11:12 | Jul 7 11:12–Jul 10 11:48 | 1,763 / 1,414 / 1,195 | 1,192 |
| 2 | Jul 1 10:00–Jul 7 11:12 | Jul 7 11:12–Jul 10 11:48 | Jul 10 11:48–Jul 13 12:24 | 3,177 / 1,195 / 1,534 | 1,529 |
| 3 | Jul 1 10:00–Jul 10 11:48 | Jul 10 11:48–Jul 13 12:24 | Jul 13 12:24–Jul 16 13:00 | 4,372 / 1,534 / 1,536 | 1,533 |

**Total static GT support is 4,254 cases.** This applies the existing GT
feature-ID check and requires `[GT anchor−300s, GT anchor+300s)` to stay in
the generation segment. A more conservative subset of 4,252 also has enough
right-side time for *any* causal detected-anchor delay up to 60 seconds. These
are upper bounds on a future same-case GT/OOS-detected RCA training cohort;
the actual OOS detected anchors, their matching, detected-anchor RCA context
legality and common cohort are **unknown**. One long event,
`gaia-v3-06443`, crosses the Jul 13 12:24 internal boundary and cannot be
counted as a complete event on either side.

Among the 4,254 GT-context-eligible candidates, labelled fault counts are
4,019 login, 212 memory, 10 file moving, 8 normal memory freed, 5 access
permission denied and **0 CPU**. The two mobile services account for
4,069/4,254 candidate roots. These are Train-side labels used only for
feasibility accounting, not detector or Test ranking features. Each fold's
full service/fault counts, exclusion counts and exact millisecond intervals
are in the machine ledger. Sparse groups require predeclared reporting and
cannot support a claim that every fault is trained or assessed equally.

An alternative four-fold design extends anchor generation through the
original C0 Validation to the 70% Train boundary. It has 5,959 static
GT-context-eligible cases, but its latter periods were used for historical
C0 checkpoint/threshold selection. It remains an **exploratory alternative**,
not an independent holdout or the frozen C1 design. The Fit-only candidate is
preferred for its cleaner historical selection boundary, not because of a
measured RCA result. The Test has already been inspected in earlier research
and cannot regain independent-confirmation status under either design. The
Fit-only periods were also part of the historical C0 model's training data;
only separately trained, forward-selected fold models could generate OOS
anchors relative to their own fitting and selection periods.

## 2. Input-fit and interface findings

| Component | Current evidence | Consequence for C1 |
|---|---|
| P6-C0 detector | Fit/Validation/Test is 50/20/30. Existing Fit scores are in-sample; Validation chose the final checkpoint/threshold. | Existing C0/C0F Fit/Validation scores, checkpoint and global threshold cannot create genuine fold OOS Train anchors. |
| Shared AD Metric | Current 48D channel includes Train-selected metric slots, quality/correlation decisions and scalers from the entire original 70% Train. | A future fold sees data-derived feature decisions from its own future if current arrays are reused. |
| Shared AD Log | Current 32D channel includes global-Train Drain3 state, stable templates and log scales. | Prefix fit needs a separate Drain3 state/schema per fold, with an explicit service/slot contract. |
| Shared AD Trace and graph | Current 8D edge channel, directed edges, scales and model graph were fitted on original 70% Train. | Per-fold edge/scale/graph fitting and trace parent-span boundary behavior need a new binding. |
| AD materializer/dataset | Low-level fit functions accept time bounds, but production materialization and dataset loading presently assume `train/test` artifacts. | A separate fold-aware writer, schema/manifest validator, reader and unique output roots are required for strict forward-OOS. |
| RCA raw index | It stores full raw time series and extracts feature values from exact case windows. Its common metric indicator set comes from the full raw filename inventory. | Decide before G2 whether the filename-derived schema is a permitted externally frozen representation; otherwise create a prefix-bound schema. Do not call it strict full-pipeline temporal isolation without this decision. |
| Existing RCA case/feature builder | Detected mode constructs matched Test cases but retains GT anchors for Train. Existing GT feature IDs support the static upper bound. | Build a new C1 Train case interface for OOS-detected anchors and a common GT/detected cohort; do not reuse the P5 detected mode as if it already did this. |

Metric/Log/Trace prefix fit primitives exist in
`src/e2e/gaia_preprocessing/raw.py`; the current train/test-only orchestration
and schema contract are in `materialize.py`, `schema.py` and `ad_data.py`.
RCA case behavior is in `scripts/p5/run_i1_rca_features.py`, with the
time-local raw index in `src/e2e/gaia_rca_adapter.py`. These are static code
facts. Whether prefix refitting produces stable valid per-fold modalities,
positive windows and enough OOS matched cases has **not** been tested.

## 3. Decision and next gate

**G1 static inventory is complete; C1 execution remains NO-GO as is.** The
Fit-only fold geometry is a candidate, not a frozen protocol. G2 must first:

1. Decide the evidence grade: strictly prefix-fitted detector inputs (and
   resolve RCA filename-schema policy), or explicitly limited
   `supervision-OOS only`. The current shared AD arrays support only the
   latter; no silent downgrade is allowed.
2. Freeze exact folds and 300-second detector-history plus separate RCA
   `[anchor−300s, anchor+300s)` boundaries. Recreate fold detector inputs
   in isolated directories if strict forward-OOS is chosen, and bind raw
   content/version, policy, schema, graph, scaler, source and completion hashes.
3. Define fold-specific checkpoint/threshold selection using only each
   preceding selection segment. Recompute Train OOS anchors; preserve
   unmatched episodes. Set a performance-blind minimum common cohort and
   sparse-group rule before any fold results.
4. Freeze the same-case GT-versus-detected RCA arms, common scaler, full
   prediction universe, label firewall, paired analysis and budgets. Only
   after these locks can a later execution run be considered.

The original C0 `BORDERLINE` verdict and C0F historical inference-source
`UNVERIFIED` limitation remain unchanged. C2 and any Test-based localization
claim remain pending.
