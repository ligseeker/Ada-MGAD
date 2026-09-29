# P6-C1-v2 G1 static feasibility report

Status: **PASS**. Read-only static audit; no model was trained and no Test label was read.

| Check | Result |
|---|---|
| Frozen Train span | 1625133600000 .. 1626963120000 (60984 bins) |
| Frozen Test start | 1626963120000 |
| Grid | exact 30 s, monotonic, int64 |
| Service order | canonical ten services, identical to shared manifest |
| Dimensions | metric 48D, log 32D, trace 8D |
| Graph | identical to the shared schema edges |
| Detector batch fields | metric, log, trace only |
| Node-anomaly label arrays | not bound, never read |
| Window legality | every fold segment hosts >= 1 legal 300 s window |
| Test isolation | no Test timestamp inside any fold segment |

## Fold geometry and reproduced static bounds

| Fold | Fit | Selection | Generation | Legal windows (F/S/G) | Static upper bound | Floor |
|---|---|---|---|---|---:|---:|
| 1 | 2021-07-01T10:00:00+00:00 | 2021-07-04T10:36:00+00:00 | 2021-07-07T11:12:00+00:00 | 8702 / 8702 / 8702 | 1192 | 596 |
| 2 | 2021-07-01T10:00:00+00:00 | 2021-07-07T11:12:00+00:00 | 2021-07-10T11:48:00+00:00 | 17414 / 8702 / 8702 | 1529 | 765 |
| 3 | 2021-07-01T10:00:00+00:00 | 2021-07-10T11:48:00+00:00 | 2021-07-13T12:24:00+00:00 | 26126 / 8702 / 8702 | 1533 | 767 |

Static upper bounds reproduce the frozen G1 ledger values (1,192 / 1,529 / 1,533). These are GT/context ceilings, not observed anchors.
The safety margin for any 0-60 s detection delay is reported per fold in `static_population.json`; it is one lower than the ceiling for folds 2 and 3.

No STOP condition triggered. The formal fold detector remains blocked until the G2/G3 gates (shared-array adapter, smoke and guards) pass.
