#!/usr/bin/env python3
"""P5-V2 audit merge step: combine the re-derived GT event registry (audit A)
with the raw-telemetry detector timeline (audit B) and produce the split /
observability / contamination artifacts required by V2 doc s13, s18, s24,
s27, s28, s41, s42, s44.

Outputs (data/p5/v2/audit/):
  split_audit.json            - 70/30 assignment of GT events under both
                                timeline calibers and both taxonomy variants,
                                boundary-crossing purge lists, same-bin
                                collision statistics, burn-in accounting.
  temporal_observability.csv  - per GT event: does the labelled service have
                                >=1 raw metric sample inside [start,end)?
                                (representative-feature approximation)
  rca_contamination.json      - per-case foreign-event counts inside the
                                W300 oracle context [t0-300s, t0+300s),
                                oracle boundary purge, clean/contaminated
                                strata sizes.

No GT count, split boundary or fault distribution is hardcoded; everything is
read from the audit artifacts.
"""

import bisect
import csv
import glob
import json
import os
from collections import Counter, defaultdict

AUDIT_DIR = "/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v2/audit"
METRIC_DIR = "/home/zhangll24/RCA_project/datasets/GAIA/MicroSS/metric/metric_split/metric"
BIN_MS = 30000
W300_MS = 300_000
WINDOW_BINS = 10  # Ada-MGAD 10x30s window; target = last bin

SERVICES = [
    "dbservice1", "dbservice2", "logservice1", "logservice2",
    "mobservice1", "mobservice2", "redisservice1", "redisservice2",
    "webservice1", "webservice2",
]


def load_events():
    primary, freed = [], []
    with open(os.path.join(AUDIT_DIR, "gt_event_registry.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ev = {
                "case_id": row["case_id"],
                "source_index": int(row["source_index"]),
                "fault_type": row["fault_type"],
                "service": row["labelled_service"],
                "start_ms": int(row["start_ms"]),
                "end_ms": int(row["end_ms"]),
                "duration_seconds": float(row["duration_seconds"]),
                "evidence_status": row["evidence_status"],
            }
            if ev["evidence_status"] == "candidate_pending_verification":
                freed.append(ev)
            else:
                primary.append(ev)
    for lst in (primary, freed):
        lst.sort(key=lambda e: (e["start_ms"], e["source_index"]))
    return primary, freed


def split_events(events, timeline_start, split_ms, timeline_end):
    train, test, crossing, outside = [], [], [], []
    for ev in events:
        if ev["end_ms"] <= timeline_start or ev["start_ms"] >= timeline_end:
            outside.append(ev)
        elif ev["start_ms"] < split_ms < ev["end_ms"]:
            crossing.append(ev)
        elif ev["end_ms"] <= split_ms:
            train.append(ev)
        else:
            test.append(ev)
    return train, test, crossing, outside


def dist(events):
    return {
        "count": len(events),
        "by_fault_type": dict(sorted(Counter(e["fault_type"] for e in events).items())),
        "by_service": dict(sorted(Counter(e["service"] for e in events).items())),
    }


def collisions(events):
    bin_all = Counter(e["start_ms"] // BIN_MS for e in events)
    bin_svc = Counter((e["service"], e["start_ms"] // BIN_MS) for e in events)
    multi = [c for c in bin_all.values() if c > 1]
    multi_svc = [c for c in bin_svc.values() if c > 1]
    return {
        "same_bin_event_bins": len(multi),
        "max_events_in_one_bin": max(multi) if multi else 0,
        "same_service_same_bin_bins": len(multi_svc),
        "max_same_service_same_bin": max(multi_svc) if multi_svc else 0,
        "definition": "events grouped by floor(start_ms/30000); positive-bin-level stats for the OLD split remain in artifacts/p5/g0r2/label_reconciliation.json",
    }


def crossing_mask(crossing, split_ms, timeline_start, timeline_end):
    """V2 s28 boundary_exclusion_mask: bins whose 10-bin AD window or label
    depends on a boundary-crossing event are excluded from supervised
    Train/Test evaluation. With strict in-split windows, the affected target
    bins are the last WINDOW_BINS-1 Train bins and the first WINDOW_BINS Test
    bins around the split, intersected with each crossing event's span."""
    affected = set()
    for ev in crossing:
        lo = max(timeline_start, ev["start_ms"] - (WINDOW_BINS - 1) * BIN_MS)
        hi = min(timeline_end, ev["end_ms"] + (WINDOW_BINS - 1) * BIN_MS)
        b = (lo // BIN_MS) * BIN_MS
        while b < hi:
            if b < split_ms < b + BIN_MS or (ev["start_ms"] < b + BIN_MS and ev["end_ms"] > b):
                if abs(b - split_ms) < WINDOW_BINS * BIN_MS:
                    affected.add(b)
            b += BIN_MS
    return sorted(affected)


def representative_metric_timestamps():
    """One representative metric feature per service per period half; returns
    {service: sorted unique epoch-ms timestamps}. Approximation for the
    observability question (V2 s18/s24): all features of a service share the
    ~30s collection cadence, but per-feature gaps differ (global gap rate
    16.18%), so absence in the representative series understates observations
    slightly; presence is exact for that feature."""
    series = {}
    for svc in SERVICES:
        stamps = []
        for half in ("2021-07-01_2021-07-15", "2021-07-15_2021-07-31"):
            files = sorted(glob.glob(os.path.join(METRIC_DIR, f"{svc}_*_{half}.csv")))
            for path in files:
                ts = []
                with open(path, encoding="utf-8") as fh:
                    head = fh.readline()  # header: timestamp,value
                    if not head.strip():
                        continue
                    for line in fh:
                        cut = line.find(",")
                        if cut > 0:
                            try:
                                ts.append(int(line[:cut]))
                            except ValueError:
                                pass
                if ts:
                    stamps.extend(ts)
                    break  # first non-empty representative file for this half
        series[svc] = sorted(set(stamps))
    return series


def observability(events, series):
    rows = []
    for ev in events:
        ts = series.get(ev["service"], [])
        i = bisect.bisect_left(ts, ev["start_ms"])
        j = bisect.bisect_left(ts, ev["end_ms"])
        rows.append({
            "case_id": ev["case_id"],
            "fault_type": ev["fault_type"],
            "service": ev["service"],
            "start_ms": ev["start_ms"],
            "end_ms": ev["end_ms"],
            "duration_seconds": ev["duration_seconds"],
            "n_metric_samples_in_interval": j - i,
            "observable": j - i > 0,
        })
    return rows


def contamination_stats(events, split_ms, timeline_start, timeline_end):
    starts = [e["start_ms"] for e in events]
    per_case = []
    for ev in events:
        t0 = ev["start_ms"]
        lo = bisect.bisect_left(starts, t0 - W300_MS - 3_600_000)
        hi = bisect.bisect_left(starts, t0 + W300_MS)
        foreign = diff_svc = diff_fault = 0
        for j in range(lo, hi):
            other = events[j]
            if other["source_index"] == ev["source_index"]:
                continue
            if other["start_ms"] < t0 + W300_MS and other["end_ms"] > t0 - W300_MS:
                foreign += 1
                if other["service"] != ev["service"]:
                    diff_svc += 1
                if other["fault_type"] != ev["fault_type"]:
                    diff_fault += 1
        ctx_start, ctx_end = t0 - W300_MS, t0 + W300_MS
        crosses_split = ctx_start < split_ms < ctx_end
        outside_timeline = ctx_start < timeline_start or ctx_end > timeline_end
        per_case.append({
            "case_id": ev["case_id"],
            "fault_type": ev["fault_type"],
            "service": ev["service"],
            "t0_ms": t0,
            "split_side": "train" if ev["end_ms"] <= split_ms else ("test" if ev["start_ms"] >= split_ms else "boundary_crossing"),
            "foreign_event_count": foreign,
            "different_service_foreign_event_count": diff_svc,
            "different_fault_foreign_event_count": diff_fault,
            "different_root_foreign_event_count": diff_svc,  # root == labelled service on GAIA
            "oracle_context_crosses_split": crosses_split,
            "oracle_context_outside_timeline": outside_timeline,
        })
    return per_case


def summarize_contamination(per_case):
    out = {}
    for side in ("train", "test"):
        rows = [c for c in per_case if c["split_side"] == side]
        counts = sorted(c["foreign_event_count"] for c in rows)
        n = len(rows)
        q = lambda p: counts[min(n - 1, int(p * n))] if n else None
        out[side] = {
            "cases": n,
            "clean_context_cases": sum(1 for c in rows if c["foreign_event_count"] == 0),
            "contaminated_context_cases": sum(1 for c in rows if c["foreign_event_count"] > 0),
            "foreign_event_count_quantiles": {"min": counts[0] if n else None, "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": counts[-1] if n else None},
            "oracle_boundary_purge": sum(1 for c in rows if c["oracle_context_crosses_split"]),
            "oracle_context_outside_timeline": sum(1 for c in rows if c["oracle_context_outside_timeline"]),
        }
    return out


def main():
    timeline = json.load(open(os.path.join(AUDIT_DIR, "detector_timeline.json"), encoding="utf-8"))
    primary, freed = load_events()
    with_freed = sorted(primary + freed, key=lambda e: (e["start_ms"], e["source_index"]))

    split_audit = {
        "gt_source": "gt_event_registry.csv (taxonomy v2-audit-20260913, re-derived from raw messages)",
        "timeline_source": "detector_timeline.json (raw telemetry, GT-independent)",
        "variants": {},
    }
    for cal_name, cal in timeline["calibers"].items():
        ts_start, ts_end, split_ms, n_bins, k = (
            cal["timeline_start_ms"], cal["timeline_end_ms"], cal["split_timestamp_ms"],
            cal["n_bins"], cal["train_bins"],
        )
        for tax_name, events in (("primary_16200_style", primary), ("with_freed", with_freed)):
            train, test, crossing, outside = split_events(events, ts_start, split_ms, ts_end)
            key = f"{cal_name}/{tax_name}"
            split_audit["variants"][key] = {
                "n_bins": n_bins,
                "train_bins": k,
                "test_bins": n_bins - k,
                "split_timestamp_ms": split_ms,
                "split_timestamp_iso": cal["split_timestamp_iso"],
                "train_events": dist(train),
                "test_events": dist(test),
                "boundary_crossing_purge": {
                    "count": len(crossing),
                    "cases": [
                        {"case_id": e["case_id"], "fault_type": e["fault_type"], "service": e["service"],
                         "start_ms": e["start_ms"], "end_ms": e["end_ms"]}
                        for e in crossing
                    ],
                },
                "outside_timeline_events": {
                    "count": len(outside),
                    "by_type": dict(Counter(e["fault_type"] for e in outside)),
                    "reason": "event interval does not intersect the telemetry timeline (e.g. injections before metric start 07-01 18:00 CST under metric_only)",
                    "cases_head": [
                        {"case_id": e["case_id"], "fault_type": e["fault_type"], "start_ms": e["start_ms"]}
                        for e in outside[:20]
                    ],
                },
                "same_bin_collisions_all": collisions(events),
                "same_bin_collisions_train": collisions(train),
                "same_bin_collisions_test": collisions(test),
                "boundary_exclusion_mask_bins": crossing_mask(crossing, split_ms, ts_start, ts_end),
                "ad_burn_in": {
                    "train_first_targets_lost": WINDOW_BINS - 1,
                    "test_first_targets_lost": WINDOW_BINS - 1,
                    "note": "strict in-split 10-bin windows (V2 s30): first 9 target bins of each split have no valid window",
                },
            }

    # ---- temporal observability (primary taxonomy, metric_only caliber) ----
    series = representative_metric_timestamps()
    obs_rows = observability(with_freed, series)
    obs_path = os.path.join(AUDIT_DIR, "temporal_observability.csv")
    with open(obs_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(obs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(obs_rows)
    by_type = defaultdict(lambda: [0, 0])
    for r in obs_rows:
        by_type[r["fault_type"]][0] += 1
        by_type[r["fault_type"]][1] += int(r["observable"])
    split_ms = timeline["calibers"]["metric_only"]["split_timestamp_ms"]
    ts_start = timeline["calibers"]["metric_only"]["timeline_start_ms"]
    obs_summary = {
        "method": "representative metric feature per service per period half; sample ts inside half-open [start_ms,end_ms)",
        "caveat": "per-feature gaps (~16% global) mean 'not observable' is slightly over-reported; 'observable' is exact for the representative feature",
        "by_fault_type": {
            t: {"events": n, "with_metric_observation": o, "share": round(o / n, 4)} for t, (n, o) in sorted(by_type.items())
        },
        "overall": {
            "events": len(obs_rows),
            "with_metric_observation": sum(r["observable"] for r in obs_rows),
        },
    }

    # ---- RCA W300 contamination (both taxonomies, metric_only caliber) ----
    cont = {}
    for tax_name, events in (("primary", primary), ("with_freed", with_freed)):
        per_case = contamination_stats(events, split_ms, ts_start, timeline["calibers"]["metric_only"]["timeline_end_ms"])
        cont[tax_name] = {
            "summary": summarize_contamination(per_case),
            "per_case_head": per_case[:5],
        }
        csv_path = os.path.join(AUDIT_DIR, f"rca_contamination_cases_{tax_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_case[0].keys()))
            writer.writeheader()
            writer.writerows(per_case)
        cont[tax_name]["per_case_csv"] = os.path.basename(csv_path)
    cont["notes"] = [
        "t0 = GT start (oracle anchor); detected-anchor (t_hat) contamination requires retraining under the new split and is N/A this audit round",
        "i1 (old 60/20/20 split) reported detected_anchor_w300_crossing_cases=0 and 1 gt_w300_ineligible case",
        "V2 s42: contaminated cases are NOT deleted; report clean/contaminated strata instead",
    ]

    out = {
        "split_audit": split_audit,
        "temporal_observability": obs_summary,
        "rca_contamination": cont,
    }
    with open(os.path.join(AUDIT_DIR, "split_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(split_audit, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(AUDIT_DIR, "rca_contamination.json"), "w", encoding="utf-8") as fh:
        json.dump(cont, fh, ensure_ascii=False, indent=2)

    v = split_audit["variants"]["metric_only/primary_16200_style"]
    print("== metric_only / primary ==")
    print("train events:", v["train_events"]["count"], v["train_events"]["by_fault_type"])
    print("test events:", v["test_events"]["count"], v["test_events"]["by_fault_type"])
    print("crossing purge:", v["boundary_crossing_purge"]["count"], [c["fault_type"] for c in v["boundary_crossing_purge"]["cases"]])
    print("outside timeline:", v["outside_timeline_events"]["count"], v["outside_timeline_events"]["by_type"])
    print("collisions all:", v["same_bin_collisions_all"])
    vf = split_audit["variants"]["metric_only/with_freed"]
    print("with_freed train/test:", vf["train_events"]["count"], vf["test_events"]["count"], "crossing:", vf["boundary_crossing_purge"]["count"])
    vu = split_audit["variants"]["union/primary_16200_style"]
    print("union/primary train/test:", vu["train_events"]["count"], vu["test_events"]["count"], "crossing:", vu["boundary_crossing_purge"]["count"], "outside:", vu["outside_timeline_events"]["count"])
    print("== observability ==")
    print(json.dumps(obs_summary["by_fault_type"], indent=1))
    print("overall:", obs_summary["overall"])
    print("== contamination (primary) ==")
    print(json.dumps(cont["primary"]["summary"], indent=1))
    cpu = [e for e in primary if e["fault_type"] == "cpu_anomalies"]
    print("cpu events:", [(e["case_id"], e["start_ms"]) for e in cpu])
    print("cpu vs split:", all(e["start_ms"] >= split_ms for e in cpu) and "ALL in Test" or "mixed")


if __name__ == "__main__":
    main()
