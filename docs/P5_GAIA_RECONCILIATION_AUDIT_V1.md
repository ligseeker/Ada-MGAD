# P5-G0R2 — GAIA Two-Stage Reconciliation Audit V1

Audit date: 2026-09-10 (Asia/Shanghai)

Scope: data/protocol audit only

Current worktree: `/home/zhangll24/RCA_project/Ada-MGAD-e2e`

Evidence directory: `artifacts/p5/g0r2/`

Evidence vocabulary in this report is binding:

- `FACT`: directly established from current code or immutable repository state.
- `REVALIDATED FACT`: historical claim reproduced from current raw data/code, or rebound to current bytes using explicit hashes.
- `HISTORICAL EVIDENCE`: useful prior asset that is not current P5 fact by itself.
- `INFERENCE`: derived interpretation whose assumptions are stated.
- `OPEN QUESTION`: evidence is insufficient to freeze semantics or protocol.
- `RECOMMENDATION`: proposed next-stage action, not an implemented/frozen method.

No ranking accuracy, F1, AC@1, Avg@5, or MRR was used to select a window, bin,
split, event subset, or method configuration.

## 1. Executive Summary

1. **REVALIDATED FACT:** The current run table contains 17,152 records. A deterministic audit-only registry identifies 16,200 supported, time-bounded injections and classifies 952 other records separately.
2. **REVALIDATED FACT:** Supported labels are extremely imbalanced: login failure is 15,478/16,200 (95.5432%); `mobservice1` and `mobservice2` jointly account for 96.3827% of labelled services.
3. **REVALIDATED FACT:** The run-table `service`, Ada-MGAD node-positive column, and historical RCA `root_service` are the same annotation by construction. The defensible term is **labelled injected service** or **labelled fault service**, not independently validated causal root cause.
4. **REVALIDATED PARSER DEFECT:** All 12 CPU injections use decimal-second duration and fail Ada-MGAD `main`'s integer-only duration regex. The audit repairs them only in its registry; production code was not changed.
5. **REVALIDATED FACT / OPEN QUESTION:** All 17 `normal memory freed label` rows have a preceding same-service memory event, but the gap from that event's end is 1,761.767–149,909.332 s (median 27,469.619 s), with 0/17 within 30 s. Their name is recovery-like; timing does not establish an immediate end marker. Taxonomy remains unfrozen.
6. **REVALIDATED FACT / OPEN SEMANTICS:** All 38 `ERROR` records pass the current non-normal label path as zero-duration events if the aligned metric timestamp exists, but official injection semantics are not established. Historical RCA excludes them.
7. **REVALIDATED FACT:** Raw half-open injection intervals yield 3,429 overlap pairs and 13,077 connected components; 4,037 events participate in overlap and 2,958/3,429 direct overlap pairs are different-root.
8. **REVALIDATED FACT:** A 30 s start grid is not event-identifying: 2,033/16,200 injections (12.5494%) share a start bin; 974 (6.0123%) share both service and start bin; 414 (2.5556%) share the complete service plus aligned positive interval. Raw injections must therefore remain the GT unit rather than being merged to fit the detector.
9. **REVALIDATED FACT:** Metrics have 30 s positive-delta median and 60 s P95; logs use a millisecond timestamp in the message prefix; traces retain microsecond start/end values and trace/span/parent identifiers.
10. **REVALIDATED FACT:** Trace codes are 200/300/400/500. Official `status != 200` marks 4,228,408 rows, while historical `status >= 500` marks only 1,156,668 and misses 3,071,740. The rules are not equivalent.
11. **REVALIDATED FACT:** Expanded W300 contexts form 322 groups (maximum 471); W600 forms only 23 (maximum 3,206). W600 telemetry activity coverage is acceptable, but grouping/contamination is severe.
12. **HISTORICAL EVIDENCE / OPEN QUESTION:** Checksum-verified historical W300 features show healthy candidate discrimination as a stage-summary proxy, but they are not Ada-RCA's exact frozen 68D representation and used the wrong historical trace-error rule. Exact health for W300/W600 × 15/30 s is not computed.
13. **INFERENCE:** A 15 s bin against nominal 30 s metrics implies a structural occupancy ceiling near 0.5 before gaps; 30 s is cadence-compatible. This does not prove 15 s is unusable and does not freeze 30 s.
14. **RECOMMENDATION:** Ada-RCA-G should build independent event-relative features from raw GAIA Metrics, Logs, and Traces, deriving four feature channels: Metric, Log, Trace Error, and Trace Latency. Ada-MGAD processed tensors are not a compatible substitute.
15. **RECOMMENDATION:** Use different split protocols: group-safe Protocol S for component RCA and contiguous chronological/purged Protocol T for continuous E2E. Always report overall, root-macro, and fault-macro results, and include detector-only service ranking as a core baseline.

## 2. Repository & Historical Asset State

State recorded before P5 edits:

| Role | Repository/worktree | Branch | HEAD | Initial working tree |
|---|---|---|---|---|
| Ada-MGAD main | `/home/zhangll24/RCA_project/Ada-MGAD` | `main` | `ab31282055b625838f41a1a3c91e51098175f254` | `?? baselines/` |
| P5 audit worktree | `/home/zhangll24/RCA_project/Ada-MGAD-e2e` | `e2e` | `ab31282055b625838f41a1a3c91e51098175f254` | clean |
| historical Ada-MGAD | `/home/zhangll24/RCA_project/Ada-MGAD-rca-standalone` | `rca-standalone` | `64bb681328fa1793014615a20ba2bc1fdf33c3b7` | clean, tracks `origin/rca-standalone` |
| Ada-RCA canonical main | `/home/zhangll24/RCA_project/Ada-RCA-cleanup` | `main` | `a2c620922e7c0ab3615d34654d4a3690d1b22c8e` | clean, tracks `origin/main` |

**FACT:** `/home/zhangll24/RCA_project/Ada-RCA` is not the main checkout; it was
`exp/p6-baselines` at `6c3abaaf...` and had unrelated untracked prediction
artifacts. It was not modified. The canonical/frozen specification used here is
`/home/zhangll24/RCA_project/Ada-RCA-cleanup/docs/REPRESENTATION_FREEZE.md`.

**FACT:** All 18 requested historical docs/source assets exist in the clean
historical worktree. Its old absolute GAIA source path under `project_2` no
longer exists. The current raw root is:

```text
/home/zhangll24/RCA_project/datasets/GAIA/MicroSS
```

The live repository capture is in `repository_state.json`. Its P5 working-tree
field reflects files created during the audit; the table above preserves the
initial state.

## 3. Revalidated GAIA Event Facts

Run table:

```text
/home/zhangll24/RCA_project/datasets/GAIA/MicroSS/run/run/run/run_table_2021-07.csv
SHA-256 ca8d44dd08319d940b0d607305165853da376208440b0e3ec07a80dac44e2820
bytes   3,642,539
```

The official GAIA repository describes `run` as containing system logs and all
anomaly-injection records. This establishes the directory's role, but not the
meaning of every message subtype.

| Measure | Count |
|---|---:|
| raw records | 17,152 |
| current-main parseable start/end | 16,243 |
| audit interpretation parseable start/end | 16,255 |
| supported injections after audit-only CPU duration interpretation | 16,200 |
| excluded/non-case records | 952 |

Level distribution: WARNING 16,217; INFO 861; ERROR 38; UNKNOWN 36.

### Supported and non-supported taxonomy

| Raw type | Count | Parse/time rule | Explicit injection? | Current P5 evidence |
|---|---:|---|---|---|
| login failure | 15,478 | WARNING text; message-prefix ms start; integer/default 11 s | yes | `REVALIDATED FACT` |
| memory anomalies | 652 | embedded `start at`; integer duration | yes | `REVALIDATED FACT` |
| CPU anomalies | 12 | embedded `start at`; raw decimal seconds | yes | `REVALIDATED PARSER DEFECT` in main; audit-only float rule |
| file moving program | 43 | embedded `start with`; integer duration | yes | `REVALIDATED FACT` |
| access permission denied exception | 15 | message-prefix ms start; one-hour/default 3,600 s | yes | `REVALIDATED FACT` |
| normal | 861 | INFO; no interval | no | `REVALIDATED FACT` as system/non-injection record |
| error event | 38 | ERROR; second-resolution prefix; `end=start` | not established | AD mapping fact; semantics open |
| normal memory freed label | 17 | WARNING; message-prefix start; main synthesizes 600 s | not established | `OPEN QUESTION` |
| unknown/unsupported | 36 | unrecognized level; no interval | no | unsupported |

The 952 excluded records are 861 normal + 38 non-injection ERROR candidates +
17 recovery-marker candidates + 36 unsupported records. This classification is
an audit registry, not a production-label rewrite.

## 4. Fault / Root Distribution

| Labelled service | Count | Percent |
|---|---:|---:|
| mobservice1 | 7,815 | 48.2407% |
| mobservice2 | 7,799 | 48.1420% |
| dbservice2 | 96 | 0.5926% |
| webservice2 | 84 | 0.5185% |
| dbservice1 | 83 | 0.5123% |
| webservice1 | 82 | 0.5062% |
| logservice1 | 66 | 0.4074% |
| logservice2 | 63 | 0.3889% |
| redisservice2 | 57 | 0.3519% |
| redisservice1 | 55 | 0.3395% |

| Fault type | Count | Percent |
|---|---:|---:|
| login failure | 15,478 | 95.5432% |
| memory anomalies | 652 | 4.0247% |
| file moving program | 43 | 0.2654% |
| access permission denied exception | 15 | 0.0926% |
| CPU anomalies | 12 | 0.0741% |

Root × fault counts:

| service | access | CPU | file moving | login | memory | total |
|---|---:|---:|---:|---:|---:|---:|
| dbservice1 | 8 | 8 | 0 | 0 | 67 | 83 |
| dbservice2 | 7 | 0 | 0 | 0 | 89 | 96 |
| logservice1 | 0 | 0 | 0 | 0 | 66 | 66 |
| logservice2 | 0 | 0 | 0 | 0 | 63 | 63 |
| mobservice1 | 0 | 0 | 0 | 7,749 | 66 | 7,815 |
| mobservice2 | 0 | 4 | 0 | 7,729 | 66 | 7,799 |
| redisservice1 | 0 | 0 | 0 | 0 | 55 | 55 |
| redisservice2 | 0 | 0 | 0 | 0 | 57 | 57 |
| webservice1 | 0 | 0 | 22 | 0 | 60 | 82 |
| webservice2 | 0 | 0 | 21 | 0 | 63 | 84 |

**INFERENCE:** Fault and root are heavily confounded. A majority-root shortcut
is 48.2407%, and a majority-fault shortcut is 95.5432% on the full registry.
This is dataset-shortcut evidence, not a trained baseline result.

## 5. AD–RCA Label Semantics

The current code path is:

```text
run-table row.service
  -> parse_anomaly_event(...).service / Event.instance
  -> deal_label label_matrix[aligned_timestamp, service] = 1
  -> label.csv / label_type.csv
  -> data_GAIA sliding-window target label[global_end - 1]
```

The historical RCA path is:

```text
run-table row
  -> historical GAIA parser
  -> RCACaseLabel.root_service = str(parsed["service"])
```

**REVALIDATED FACT:** There is no additional causal inference in the historical
adapter. For the 16,188 supported injections that current main can map, AD
positive service equals historical RCA labelled service in 16,188/16,188 cases
(100%). If the 12 CPU parser failures are treated as mismatches, the ratio is
16,188/16,200 (99.9259%).

The equality is annotation coupling, not two independent labels. The report
therefore uses `labelled injected service` or `labelled fault service`.

At supported-injection positive bins:

- 30,907 distinct positive time bins are covered.
- Injection multiplicity per bin: 1 → 25,040; 2 → 5,194; 3 → 619; 4 → 50; 5 → 4.
- Distinct positive services per bin: 1 → 26,070; 2 → 4,463; 3 → 354; 4 → 19; 5 → 1.
- Same-root multi-injection bins: 1,030; different-root multi-injection bins: 4,837.

### Label reconciliation table

| raw type | raw semantics | current AD label | historical RCA eligible | candidate E2E meaning | evidence/status |
|---|---|---|---|---|---|
| login failure | explicit simulated failure | event service positive on aligned interval | yes | labelled injected service | raw + code, `REVALIDATED FACT` |
| memory anomalies | explicit high-memory injection | event service positive | yes | labelled injected service | raw + code, `REVALIDATED FACT` |
| CPU anomalies | explicit CPU injection, decimal duration | absent under current integer regex | yes after historical/audit repair | labelled injected service after parser decision | `REVALIDATED PARSER DEFECT` |
| file moving program | explicit trigger | event service positive | yes | labelled fault service | `REVALIDATED FACT` |
| access permission denied exception | explicit exception injection | event service positive | yes | labelled fault service | `REVALIDATED FACT` |
| normal | INFO/system record | no positive label | no | normal/non-event | `REVALIDATED FACT` |
| error event | generic system error | one aligned service-positive bin if timestamp exists | no | unresolved observation, not established injection | mapping fact; semantics open |
| normal memory freed label | normal memory-free marker name | main synthesizes a 600 s positive interval | no | recovery candidate | `OPEN QUESTION` |
| unknown | unsupported | no interval/positive label | no | unsupported/open | `REVALIDATED FACT` |

Final taxonomy is deliberately **NOT FROZEN** in this round.

## 6. CPU / Recovery / Error Record Audit

### CPU

All 12 CPU records contain decimal duration (examples 2.892–3.241 s). Current
main parses 0/12 complete intervals; the audit-only decimal interpretation
parses 12/12. Status: **REVALIDATED PARSER DEFECT**. No production fix was made.

### `normal memory freed label`

All 17 have an earlier memory event on the same service. However:

- minimum gap after the earlier memory interval: 1,761.767 s;
- median: 27,469.619 s;
- maximum: 149,909.332 s;
- within 30 s of the earlier end: 0/17.

The literal name and historical exclusion are recovery evidence; current timing
does not support treating the row as the immediate end of the nearest prior
memory injection. Whether it is a delayed/global recovery marker is an **OPEN
QUESTION**. It must not silently become either a standalone fault or a deleted
record.

### ERROR

Current parsing produces 38 valid zero-duration `error_event` intervals. The
label filter excludes only `normal`, so all 38 can enter `label.csv` at one
aligned bin when that bin exists in the metric index. Historical RCA excludes
all 38 as non-injection records. Mapping is a `REVALIDATED FACT`; injection
semantics are an `OPEN QUESTION`.

## 7. Injection Density & 30s Resolution

Raw interval overlap uses only `[start_time,end_time)`, with no ±context:

| Measure | Value |
|---|---:|
| isolated events | 12,163 |
| same-root-only overlap events | 567 |
| different-root overlap events | 3,470 |
| overlap pairs | 3,429 |
| same-root pairs | 471 |
| different-root pairs | 2,958 |
| connected components | 13,077 |
| maximum component | 39 |

Global consecutive-start gap median is 106.730 s; P10/P25/P75/P90 are
21.706/48.588/210.728/355.340 s. Fractions below 15/30/60/120/300 s are
5.7411%/15.1120%/30.7426%/54.3922%/86.0177%.

Same-root pooled median is 234.953 s; the corresponding fractions are
2.6436%/7.2020%/15.3984%/29.8579%/58.6041%. Global adjacent different-root
median is 108.335 s; fractions are 5.8114%/15.0454%/30.5272%/53.9221%/85.6667%.

Fault-specific consecutive median (s): login 111.755; memory 2,995.820; file
moving 30,599.907; CPU 13,411.553; access 86,369.759. Full percentiles and
thresholds are in `temporal_resolution.json`.

### 30 s start-bin collision

Across all 89,280 July calendar bins: 0 injections 83.0085%; 1 injection
15.8681%; 2 injections 1.0932%; ≥3 injections 0.0302%. Among the 15,170
event-bearing bins, 1,003 are collisions: 467 same-root only, 520 different-root
only, and 16 mixed.

Resolution has several defensible denominators:

- **strict time-anchor collision:** 2,033/16,200 = 12.5494% share a 30 s start bin;
- **same node-time onset lower bound:** 974/16,200 = 6.0123% share service and start bin;
- **any shared same-service positive bin:** 2,151/16,200 = 13.2778%; this does not imply their full interval signatures are identical;
- **fully identical service + aligned interval:** 414/16,200 = 2.5556%.

The main answer to “can a 30 s output uniquely identify every injection time?”
is **NO; 12.5494% have non-unique start bins**. The 6.0123% value is the hard
same-node onset collision bound. These are label-resolution limits, not reasons
to merge ground truth.

One injection covers a mean 2.315 aligned AD bins (median 1; maximum 121) under
current inclusive-end implementation. 15,490/16,200 (95.6173%) have raw
duration below both 30 and 60 s. Short duration alone does **not** imply an event
is indistinguishable: a unique short injection may still own a unique positive
bin.

Resolution-compatible descriptive subsets: nearest other injection >30 s:
11,660 (71.9753%); >60 s: 7,758 (47.8889%); >120 s: 3,354 (20.7037%). These are
not filtered cohorts.

## 8. Raw vs Expanded Context Overlap

Expanded contexts include the raw interval union the half-open anchor context.
They must not be confused with raw interval density.

| Audit | Groups | Non-singleton groups | Cases in non-singletons | Largest | Direct overlap pairs | Different-root pair ratio |
|---|---:|---:|---:|---:|---:|---:|
| raw injection intervals | 13,077 | 914 | 4,037 | 39 | 3,429 | 86.2642% |
| W300 expanded | 322 | 306 | 16,184 | 471 | 64,956 | 54.6401% |
| W600 expanded | 23 | 18 | 16,195 | 3,206 | 127,820 | 53.7279% |

W300 still has 299 multi-root groups covering 16,160 cases; it is a relative
improvement, not contamination-free. W600's 23 components create a particularly
restrictive group-safe split problem.

## 9. Ada-RCA Temporal Compatibility

**FACT:** Frozen Ada-RCA uses `[t0-600s,t0+600s)`, 15 s bins, 80 bins, event-local
pre-event median/MAD with IQR fallback, Q90 service/channel aggregation, explicit
masks, and 68D Z2 features. Raw modalities are Metrics, Logs, Traces; derived
feature channels are Metric, Log, Trace Error, Trace Latency.

Historical full-scan activity proxies rebound to the current layout show:

| Window | Metrics any-case activity | Logs | Traces |
|---|---:|---:|---:|
| W300 | 88.6296% | 99.9753% | 99.9753% |
| W600 | 88.7716% | 99.9753% | 99.9753% |

These historical coverage values use inclusive 30 s occupancy and are not exact
frozen q-mask coverage. W600 is therefore coverage-compatible but group-hostile.
W300 has a clear component/split feasibility advantage with nearly identical
activity coverage, but its final choice remains **NOT FROZEN**.

For bins, 15 s preserves the standalone specification but nominal 30 s metrics
imply an upper occupancy reference of 0.5 (gap-adjusted 0.4191). At 30 s the
references are 1.0 and 0.8382. These are cadence-derived expectations, not
measured exact 68D health. 15 s is not proven structurally all-missing; 30 s is
more cadence-aligned but changes the frozen temporal semantics. Bin size remains
**NOT FROZEN**.

## 10. Representation Health

An exact GAIA four-channel materializer was not frozen in this round, so no
exact 68D representation was built for the four requested configurations. This
is a deliberate evidence boundary, not a successful result omitted from the
report.

The checksum-verified historical W300 stage-summary bundle provides only a
non-68D proxy:

| Feature channel | available proxy | observed-cell coverage | active proxy | all-zero | all-masked | constant-feature ratio |
|---|---:|---:|---:|---:|---:|---:|
| Metric | 100.0000% | 85.6564% | 83.7439% | 16.2561% | 0% | 0% |
| Log | 99.9777% | 97.8731% | 96.5026% | 3.4974% | 0.0223% | 0% |
| Trace Error | 92.6949% | 92.6949% | 70.2710% | 29.7290% | 7.3051% | 0% |
| Trace Latency | 92.6949% | 92.6949% | 92.6949% | 7.3051% | 7.3051% | 0% |

All observed proxy cells are finite. Ten candidates are effective per proxy
case. All candidates have identical signatures in 0.00742% of cases; any
identical candidate pair occurs in 0.05197%; mean unique-signature discrimination
is 0.999651.

Limitations are material: the proxy uses a historical 13,470-case cohort,
stage-summary rather than 15/30 s Z2 bins, no W600 array, and the invalid
historical `status >= 500` trace-error definition. Therefore the exact four-way
representation-health requirement remains an **OPEN QUESTION** for the next
adapter-freeze round.

## 11. Raw-vs-Processed RCA Input Decision

### Option A — reuse Ada-MGAD processed representation

**RECOMMENDATION: do not use.** The 30 s detector representation aggregates or
changes raw timing, log text/schema, trace status/latency, and trace structure.
It does not satisfy Ada-RCA's four derived numeric-channel contract and would
unnecessarily couple diagnosis features to detector preprocessing.

### Option B — independent raw event-relative preprocessing

**RECOMMENDATION: use for Ada-RCA-G.** Minimum adapter work is:

1. Map long metric shards to label-free `<candidate>_<indicator>` series.
2. Parse raw log message-prefix timestamps and create numeric log indicators.
3. Derive Trace Error using GAIA `status != 200`.
4. Derive Trace Latency from `end_time-start_time`.
5. Use one explicit numeric time unit for all channels and the event anchor.
6. Preserve masks, source hashes, aggregation rules, candidate registry, and a label firewall.

This is a data/time adapter recommendation. It does not change Ada-RCA 68D,
Conditional Logit, loss, or training method.

## 12. Split Feasibility

### Protocol S — component/integration

The audit constructed deterministic five-fold grouped-stratified candidates;
expanded context groups never cross folds.

| Window | Groups | Largest | Fold event counts | All roots/faults in every fold? | Feasible candidate? |
|---|---:|---:|---|---|---|
| W300 | 322 | 471 | 3,241 / 3,240 / 3,238 / 3,239 / 3,242 | yes | yes |
| W600 | 23 | 3,206 | 3,212 / 3,429 / 3,025 / 3,371 / 3,163 | no; CPU absent in 4/5 folds, access absent in 1 | no under this requirement |

This candidate controls component distribution and context leakage. It does not
measure temporal deployment generalization. No assignment is frozen yet;
training-only preprocessing remains mandatory.

### Protocol T — continuous temporal E2E

Both candidates are chronological contiguous streams; no scattered event
windows are concatenated.

| Candidate | Train/val/test events | CPU placement | Access placement | W300 crossing | W600 crossing | AD 10×30 s crossing |
|---|---|---|---|---:|---:|---:|
| duration 60/20/20 | 8,787 / 3,513 / 3,900 | 0 / 0 / 12 | 8 / 3 / 4 | 2 | 3 | 1 |
| event-count 60/20/20 | 9,720 / 3,240 / 3,240 | 0 / 0 / 12 | 10 / 1 / 4 | 7 | 18 | 4 |

For duration blocks, durations are 1,583,028.038 / 527,676.013 / 527,676.013
s. For event-count blocks they are 1,754,803.767 / 450,476.045 / 433,100.252
s. Purging crossing expanded contexts retains 16,198/16,197 events at W300/W600
for duration splits and 16,193/16,182 for event-count splits.

CPU appears only on July 27, 28, 29, and 31 (1/6/3/2 cases). All 12 land in the
test block under both natural candidates. That repeated outcome indicates a
real temporal distribution shift, not merely an artifact of one chosen cut.
Missing CPU from train/validation should be reported, not used to reject a
continuous temporal evaluation. Exact Protocol T boundaries remain **NOT
FROZEN**.

## 13. Frequency Shortcut Risk

The majority-root shortcut is 48.2407%; the two mobile services comprise
96.3827%. The majority-fault shortcut is 95.5432%, and root × fault support is
nearly deterministic for several rare faults. Consequently:

- **RECOMMENDATION:** report overall, root-macro, and fault-macro for RCA.
- **RECOMMENDATION:** include a detector-only service ranking baseline. Because
  AD and RCA service labels are annotation-coupled, this baseline is essential
  to expose whether apparent E2E localization merely repeats detector node
  scores or dataset frequency.
- Historical trained baseline scores remain historical context only and were
  not rerun or used for configuration selection.

## 14. Historical Asset Reuse Matrix

The machine-readable matrix is `historical_reuse_matrix.csv`. Summary:

| Historical asset | Revalidated now? | P5 use |
|---|---|---|
| DATASET_AUDIT | partial; inventory claims revalidated | parser/taxonomy logic and expected counts |
| GAIA_INCLUSION_AUDIT | no current freeze | inclusion/purity semantics only; old 13,470/2,730 cohorts not inherited |
| TELEMETRY_DIAGNOSTICS | rebound by run hash + layout digests; trace status rescanned | timing/coverage evidence without repeating the full historical scan |
| SPLIT_DIAGNOSTICS | logic reused, assignments not inherited | context grouping and validation design |
| P2_METRIC_FEATURES | timing/coverage premise partially revalidated | representation-health prior only |
| P2_MODALITY_SCHEMA_AUDIT | raw schema rechecked; feature status conflicted | channel schema reference, not canonical P5 features |
| BASELINE_RESULTS | not revalidated as model evidence | frequency-shortcut context only |
| protocol/status docs | readable historical design record | conservative terminology and evidence levels |

Historical facts revalidated: 17,152/16,200/952 inventory, CPU float issue,
root/fault imbalance, raw overlap, W300/W600 group counts, raw schemas/timing,
and current-byte telemetry binding.

Historical conclusions denied or downgraded:

- `status >= 500` is **denied** for current GAIA Trace Error semantics.
- Immediate recovery/end-marker timing for `normal memory freed label` is **not supported**.
- Old absolute source paths and `Verification Status: UNVERIFIED` documents are historical only.
- Historical 13,470/2,730 inclusion cohorts and split assignments are not P5 freezes.
- P2 arrays are not exact Ada-RCA 68D or four-configuration health evidence.

## 15. Open Questions

1. What official semantics, if any, distinguish the 17 memory-freed rows from generic system/recovery logs?
2. Should the 38 generic ERROR rows be excluded from final AD supervision, represented as observations only, or treated as a separate detection taxonomy?
3. After a label-policy decision, should production CPU parsing be repaired and all AD data regenerated?
4. What exact mask/variance/discrimination health results arise from an audit-only raw GAIA four-channel materializer for W300/W600 × 15/30 s?
5. Which chronological boundary/purge policy will be preregistered for Protocol T, accepting late-only CPU as distribution shift?

## 16. Risks

1. **Annotation-coupling risk:** AD and RCA service labels share the same source field, so localization gains may not constitute independent causal evidence.
2. **Shortcut/confounding risk:** extreme root/fault imbalance and restricted root × fault support can inflate overall scores.
3. **Context-dependence risk:** W600 produces only 23 groups, making leakage-safe distribution control fragile.
4. **Representation risk:** exact four-configuration 68D health is not yet available; historical proxies have incompatible trace-error semantics.
5. **Temporal-shift risk:** CPU occurs only late in July; chronological training may contain no CPU examples, which limits claims but is a real deployment-style condition.

## 17. Recommended P5 Research Design

1. Preserve each of the 16,200 audit-supported raw injections as a canonical GT
   registry unit; do not merge events to fit 30 s detector outputs.
2. Before formal training, freeze the CPU repair and explicit policies for
   recovery/ERROR rows. Regenerate Ada-MGAD-G only if that frozen label contract
   changes training labels or processed data.
3. Treat Ada-MGAD output solely as trigger evidence. Do not presuppose detector
   service scores as Ada-RCA inputs.
4. Build an independent, label-free raw GAIA adapter that materializes the four
   frozen feature channels while leaving 68D and Conditional Logit unchanged.
5. Retain W600+B15 as the frozen standalone reference. Carry W300 and B30 as
   compatibility candidates; W300 has strong group/split evidence, and B30 has
   cadence evidence, but neither is frozen until exact representation health is
   available.
6. Use Protocol S for component/oracle RCA and Protocol T for continuous E2E;
   do not force one split to answer both questions.
7. Preregister overall/root-macro/fault-macro metrics and detector-only service
   ranking before any GAIA model-performance inspection.

## Q1–Q20 Answers

| Q | Answer | Concise conclusion |
|---|---|---|
| Q1 | **YES** | 16,200 stable, supported injection records are reproducibly identifiable from 17,152 raw rows. |
| Q2 | **YES** | `service` is supported as labelled injected/fault service, not independently validated causal root cause. |
| Q3 | **YES** | Current AD positive-node column directly uses the same parsed service for mappable events. |
| Q4 | **CONDITIONAL** | 100% among 16,188 main-mappable injections; 99.9259% over all 16,200 if 12 missing CPU rows count as mismatch. Semantics are coupled by construction. |
| Q5 | **YES** | Float-duration defect reproduces for 12/12 CPU rows. |
| Q6 | **UNKNOWN** | Name/history suggest recovery, but nearest same-service timing is far too delayed to establish an immediate end marker. |
| Q7 | **YES** | 38 ERROR rows pass the current AD label path, conditional on aligned metric timestamps; official injection meaning is unresolved. |
| Q8 | **CONDITIONAL** | 12.5494% have non-unique 30 s start bins; hard same-node onset collision is 6.0123%; fully identical aligned service intervals are 2.5556%. |
| Q9 | **YES** | Keep raw injections as GT; never merge official units merely to match detector resolution. |
| Q10 | **CONDITIONAL** | W600 activity coverage is acceptable, but 23 groups/max 3,206 makes contamination and group-safe splitting severe; exact 68D health is open. |
| Q11 | **YES, CONDITIONAL** | W300 has clear group/split compatibility advantage (322 groups/max 471) with similar activity coverage, not proven performance advantage. |
| Q12 | **NO** | Structural all-missing/degenerate 68D is not proven. A 15 s metric occupancy ceiling near 0.5 is an important risk. |
| Q13 | **CONDITIONAL** | 30 s matches nominal metric cadence better, but changes frozen 15 s semantics and needs exact health audit. |
| Q14 | **status != 200** | Official semantics require it; codes 300/400 make `>=500` non-equivalent and cause 3,071,740 missed error rows. |
| Q15 | **YES** | Use independent raw event-relative preprocessing, not Ada-MGAD processed representation. |
| Q16 | **YES** | Protocol S answers controlled component integration; Protocol T answers chronological continuous E2E. |
| Q17 | **YES** | Overall + root-macro + fault-macro are necessary under the observed imbalance/confounding. |
| Q18 | **YES** | Detector-only service ranking must be a core baseline to expose annotation coupling and shortcut behavior. |
| Q19 | **CONDITIONAL** | Do not retrain now. Retrain Ada-MGAD-G only after taxonomy/parser policy is frozen and only if resulting supervision/preprocessing changes. |
| Q20 | **CONDITIONAL** | Minimum: raw GAIA four-channel adapter, explicit units/timestamps, `!=200` Trace Error, latency derivation, masks/provenance, context/split handling. Do not modify 68D, Conditional Logit, loss, model, or use performance to select configuration. |

# P5 Decision Table

| Decision | Evidence | Status | Next action |
|---|---|---|---|
| canonical event registry | current run hash; 16,200 supported, 952 classified non-cases | READY AS AUDIT EVIDENCE; **NOT FROZEN** | review taxonomy and freeze next round |
| GT injection unit | raw records are stable; 30 s collisions are measurable | **RECOMMENDED: raw injection** | freeze without diagnosis-episode merging |
| AD label taxonomy | CPU missing; recovery/ERROR semantic boundary differs from RCA | **NOT FROZEN** | decide CPU/recovery/ERROR policy before data regeneration |
| CPU parser | 12/12 decimal durations fail main | **REVALIDATED PARSER DEFECT** | make a separate reviewed production fix only after protocol freeze |
| recovery marker | name/history support recovery, timing does not support immediate end | **NOT FROZEN** | seek official semantics or preregister conservative sensitivity handling |
| ERROR records | 38 map to AD; historical RCA excludes; injection meaning absent | **NOT FROZEN** | freeze explicit AD/RCA treatment |
| W600 | activity coverage acceptable; 23 groups/max 3,206; Protocol S rare-fault failure | **NOT FROZEN / HIGH RISK** | retain as standalone reference and compatibility sensitivity |
| W300 | 322 groups/max 471; feasible five-fold candidate; similar coverage | **NOT FROZEN / PREFERRED CANDIDATE** | exact four-channel health audit, then preregister |
| 15 s bin | frozen standalone reference; cadence occupancy risk | **NOT FROZEN FOR GAIA** | retain reference; measure exact masks/variance |
| 30 s bin | nominal metric cadence aligned; semantics differ from standalone | **NOT FROZEN / PREFERRED CANDIDATE** | exact four-channel health audit, then preregister |
| raw RCA preprocessing | raw schemas preserve logs/status/latency/parent spans | **RECOMMENDED** | implement independent adapter next round; leave core extractor/model unchanged |
| component split | W300 group-safe five-fold candidate covers all roots/faults | **NOT FROZEN** | use Protocol S candidate after manifest/preprocessing freeze |
| continuous E2E split | two contiguous candidates audited; CPU late-only; small explicit purge | **NOT FROZEN** | preregister Protocol T boundary and purge policy |
| detector-only baseline | service labels coupled and detector emits node evidence | **REQUIRED RECOMMENDATION** | define before model evaluation |
| macro metrics | majority root 48.24%, majority fault 95.54%, root×fault confounding | **REQUIRED RECOMMENDATION** | preregister overall/root-macro/fault-macro |
| Ada-MGAD retraining | current label contract is unresolved | **DO NOT RUN NOW** | retrain only if frozen label/preprocessing changes require it |
| Ada-RCA retraining | no GAIA adapter/config/splits frozen | **DO NOT RUN NOW** | first freeze minimal data/time adapter; do not change core method |

## Reproduction

```bash
python scripts/p5/revalidate_gaia_events.py \
  --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --output-dir artifacts/p5/g0r2

python scripts/p5/audit_label_reconciliation.py \
  --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --output-dir artifacts/p5/g0r2

python scripts/p5/audit_injection_resolution.py \
  --event-registry artifacts/p5/g0r2/event_registry.csv \
  --output artifacts/p5/g0r2/temporal_resolution.json

python scripts/p5/audit_raw_telemetry.py \
  --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS \
  --historical-telemetry /home/zhangll24/RCA_project/Ada-MGAD-rca-standalone/artifacts/p1/telemetry_diagnostics.json \
  --scan-traces \
  --output artifacts/p5/g0r2/raw_telemetry_timing.json

python scripts/p5/audit_rca_compatibility.py \
  --event-registry artifacts/p5/g0r2/event_registry.csv \
  --raw-telemetry artifacts/p5/g0r2/raw_telemetry_timing.json \
  --ada-rca-repo /home/zhangll24/RCA_project/Ada-RCA-cleanup \
  --historical-repo /home/zhangll24/RCA_project/Ada-MGAD-rca-standalone \
  --output artifacts/p5/g0r2/rca_compatibility.json

python scripts/p5/audit_p5_split_feasibility.py \
  --event-registry artifacts/p5/g0r2/event_registry.csv \
  --output artifacts/p5/g0r2/split_feasibility.json

python scripts/p5/build_historical_reuse_and_ledger.py \
  --output-dir artifacts/p5/g0r2 \
  --historical-root /home/zhangll24/RCA_project/Ada-MGAD-rca-standalone \
  --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS
```

The full trace status/duration command reads all trace rows. The metrics/logs
full-scan facts were not needlessly recomputed: they were upgraded only after
the current run-table SHA and all three relative-path/byte-size layout digests
matched the historical artifact exactly. Trace status was rescanned because its
semantic rule directly changes the required RCA feature channel.

## Scope Closure

本轮没有训练正式 Ada-MGAD / Ada-RCA，没有修改两种方法的核心模型或训练目标，
没有实现最终两阶段 pipeline，也没有根据测试性能选择 GAIA-specific 方法配置。
