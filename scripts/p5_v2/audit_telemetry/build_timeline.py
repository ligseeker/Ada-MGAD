#!/usr/bin/env python3
"""P5-V2 audit B: build the GT-independent 30s detector timeline and the
chronological 70/30 split boundary (V2 doc sections 12-14).

The timeline endpoints below were established by a full raw-telemetry scan
during this audit round and are cross-verified against committed evidence:
  * metric endpoints: first sample 1625133601000 (2021-07-01 18:00:01 CST,
    head of *2021-07-01_2021-07-15.csv files), latest sample 1627747200000
    (2021-08-01 00:00:00 CST, tail of 3 redis_keyspace half-2 files;
    spot-checked tail of a late dbservice1 file: 1627747170000).
  * logs/traces earliest sample 1625104624255 (2021-07-01 09:57:04.255 CST,
    webservice1 trace / business), floored to 1625104620000 for the grid.
  * trace/business full counts match artifacts/p5/g0r2/raw_telemetry_timing.json
    (committed historical full scan) key-for-key.

Two calibers are reported:
  metric_only  - grid spans the metric telemetry range. This matches the
                 original Ada-MGAD main timeline definition (pre_GAIA.py
                 builds the 30s grid from aligned metric timestamps), so it
                 is the RECOMMENDED primary caliber.
  union        - grid spans all three modalities (logs/traces start ~8h
                 earlier than metrics).

For each caliber: N bins, K = floor(0.7*N), split_timestamp = start of bin K,
Train = bins [0,K), Test = bins [K,N). Nothing here depends on the GT
taxonomy; changing GT inclusion rules cannot move the boundary.
"""

import json
import os

OUT_DIR = "/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v2/audit"
BIN_MS = 30000
TRAIN_RATIO = 0.7

CALIBERS = {
    "metric_only": {
        "timeline_start_ms": 1625133600000,  # floor30s(2021-07-01 18:00:01 CST)
        "timeline_end_ms": 1627747200000,    # exclusive end = latest metric sample (2021-08-01 00:00:00 CST)
        "first_sample_ms": 1625133601000,
        "last_sample_ms": 1627747200000,
    },
    "union": {
        "timeline_start_ms": 1625104620000,  # floor30s(2021-07-01 09:57:00 CST)
        "timeline_end_ms": 1627747200000,
        "first_sample_ms": 1625104624255,    # webservice1 trace 09:57:04.255
        "last_sample_ms": 1627747199906,     # trace 07-31 23:59:59.906
    },
}


def iso(ms):
    from datetime import datetime, timedelta, timezone
    cst = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(ms / 1000, cst).isoformat(timespec="milliseconds")


def build(name, c):
    start, end = c["timeline_start_ms"], c["timeline_end_ms"]
    assert start % BIN_MS == 0 and end % BIN_MS == 0
    n_bins = (end - start) // BIN_MS
    # exact integer arithmetic: floor(0.7*N) == (7*N)//10
    # (float 0.7*N would truncate 60983.999... to 60983 for N=87120)
    k = (7 * n_bins) // 10
    split_ms = start + k * BIN_MS
    # sensitivity: if the exclusive end were extended by one bin (samples
    # exactly at 1627747200000 counted into a final bin), does K/split move?
    n_alt = n_bins + 1
    k_alt = (7 * n_alt) // 10
    return {
        "caliber": name,
        "timeline_start_ms": start,
        "timeline_start_iso": iso(start),
        "timeline_end_ms": end,
        "timeline_end_iso": iso(end),
        "first_raw_sample_ms": c["first_sample_ms"],
        "last_raw_sample_ms": c["last_sample_ms"],
        "bin_width_ms": BIN_MS,
        "n_bins": n_bins,
        "train_bins": k,
        "test_bins": n_bins - k,
        "k_floor_0_7_n": k,
        "split_timestamp_ms": split_ms,
        "split_timestamp_iso": iso(split_ms),
        "actual_train_ratio": round(k / n_bins, 6),
        "actual_test_ratio": round((n_bins - k) / n_bins, 6),
        "sensitivity_n_plus_1": {
            "n_bins": n_alt,
            "k": k_alt,
            "split_timestamp_ms": start + k_alt * BIN_MS,
            "split_unchanged": k_alt == k,
        },
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "definition": "B_k = [timeline_start + k*30s, +30s); Train = B_0..B_(K-1); Test = B_K..B_(N-1); K = floor(0.7*N); T_split = start(B_K)",
        "gt_dependence": "none - endpoints come from raw telemetry only (V2 s12/s13)",
        "primary_recommendation": "metric_only (matches Ada-MGAD main pre_GAIA.py metric-timestamp grid definition)",
        "calibers": {name: build(name, c) for name, c in CALIBERS.items()},
        "provenance": {
            "endpoints_from": "full raw-telemetry scan, audit round 2026-09-13",
            "cross_verified_against": [
                "artifacts/p5/g0r2/raw_telemetry_timing.json (committed full scan; trace/business counts match)",
                "head/tail spot check of metric half-1/half-2 files (this audit round)",
            ],
            "rescan_instructions": "endpoints can be re-derived via scripts/p5/audit_raw_telemetry.py (G0R2) or head/tail over metric/metric_split/metric, business/business_split/business, trace/trace_split/trace",
        },
    }
    path = os.path.join(OUT_DIR, "detector_timeline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out["calibers"], indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
