#!/usr/bin/env python3
"""P5-V2 audit A: re-derive the GAIA GT taxonomy from raw run-table message text.

Input : .../run/run/run/run_table_2021-07_fixed.csv
        Columns datetime,service,message are the preserved raw fields; the
        message column holds the full raw log line
        "<ts>,<ms> | LEVEL | node_ip | container_ip | service | payload".
        Columns timestamp/level/service_node/anomaly_type/duration_seconds are
        the owner's pre-parse CLAIMS: everything is re-derived from the raw
        message and then cross-checked against those columns.

Output: data/p5/v2/audit/{run_record_registry.csv, gt_event_registry.csv,
        taxonomy_summary.json, normal_memory_freed_audit.json,
        error_records_audit.json, cpu_anomaly_audit.json,
        preparse_crosscheck.json[, preparse_mismatches.csv]}

N_GT is an output of this script, never an input. No expected count is
hardcoded or asserted. Stdlib only; timezone fixed to Asia/Shanghai (UTC+8,
no DST in 2021).
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

RUN_TABLE = "/home/zhangll24/RCA_project/datasets/GAIA/MicroSS/run/run/run/run_table_2021-07_fixed.csv"
OUT_DIR = "/home/zhangll24/RCA_project/Ada-MGAD-e2e-v2/data/p5/v2/audit"
CST = timezone(timedelta(hours=8))
TAXONOMY_VERSION = "v2-audit-20260913"
BIN_MS = 30000
# Audit B (metric-only detector timeline start, 2021-07-01 18:00:00 CST).
METRIC_TIMELINE_START_MS = 1625133600000

SUPPORTED_FAULT_TYPES = (
    "login_failure",
    "memory_anomalies",
    "file_moving",
    "access_permission_denied",
    "cpu_anomalies",
)

WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}

TS_MSG_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})$")
TS_EMBEDDED_RE = re.compile(
    r"start (?:at|with)\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)
NUM_SECONDS_RE = re.compile(
    r"(?:lasts|last for|wait for)\s+(\d+(?:\.\d+)?)\s+seconds"
)
WORD_SECONDS_RE = re.compile(r"(?:lasts|last for|wait for)\s+([a-z]+)\s+seconds")
WORD_MINUTES_RE = re.compile(r"lasts\s+([a-z]+)\s+minutes")
NUM_MINUTES_RE = re.compile(r"lasts\s+(\d+(?:\.\d+)?)\s+minutes")
HOUR_RE = re.compile(r"lasts?\s+an?\s+hour")

NORMALIZE_RES = (
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"), "<TS>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<N>"),
)


def epoch_ms(dt):
    return int(round(dt.replace(tzinfo=CST).timestamp() * 1000))


def iso_ms(ms):
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, CST).isoformat(timespec="milliseconds")


def parse_msg_ts(field):
    m = TS_MSG_RE.match(field.strip())
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return epoch_ms(dt) + int(m.group(2))


def parse_embedded_ts(text):
    m = TS_EMBEDDED_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return epoch_ms(datetime.strptime(raw, fmt))
        except ValueError:
            continue
    return None


def parse_duration_seconds(payload):
    """Return (seconds|None, how) from raw payload text; supports decimal
    seconds (\d+(?:\.\d+)?), english word seconds/minutes and 'an hour'."""
    m = NUM_SECONDS_RE.search(payload)
    if m:
        return float(m.group(1)), "numeric_seconds"
    m = WORD_SECONDS_RE.search(payload)
    if m and m.group(1) in WORD_NUM:
        return float(WORD_NUM[m.group(1)]), "word_seconds:" + m.group(1)
    m = WORD_MINUTES_RE.search(payload)
    if m and m.group(1) in WORD_NUM:
        return float(WORD_NUM[m.group(1)] * 60), "word_minutes:" + m.group(1)
    m = NUM_MINUTES_RE.search(payload)
    if m:
        return float(m.group(1)) * 60, "numeric_minutes"
    if HOUR_RE.search(payload):
        return 3600.0, "word_hour"
    return None, None


def classify(payload, level):
    if "[memory_anomalies]" in payload:
        return "memory_anomalies"
    if "[cpu_anomalies]" in payload:
        return "cpu_anomalies"
    if "[normal memory freed label]" in payload:
        return "normal_memory_freed"
    low = payload.lower()
    if "login failure" in low:
        return "login_failure"
    if "file moving program" in low:
        return "file_moving"
    if "access permission denied" in low:
        return "access_permission_denied"
    if "upload" in low and "failed" in low:
        return "log_upload_failure"
    if "get a error" in low or "object is not callable" in low:
        return "exception_injection_failed"
    if "[normal]" in low:
        return "normal"
    if "upload" in low and "successfully" in low:
        return "upload_success"
    return "unknown"


# start semantics per raw_type: "embedded" = payload 'start at/with' field,
# "message_ts" = the raw log line prefix timestamp.
START_SEMANTICS = {
    "memory_anomalies": "embedded",
    "cpu_anomalies": "embedded",
    "file_moving": "embedded",
    "login_failure": "message_ts",
    "access_permission_denied": "message_ts",
    "normal_memory_freed": "message_ts",
}
DEFAULT_DURATION = {  # seconds; only used when payload carries no duration
    "login_failure": 11.0,
    "access_permission_denied": 3600.0,
    "normal_memory_freed": 600.0,
}


def normalize_pattern(payload):
    text = payload
    for rx, rep in NORMALIZE_RES:
        text = rx.sub(rep, text)
    return re.sub(r"\s+", " ", text).strip()[:160]


def norm_type_token(value):
    return re.sub(r"[\s\[\]]+", "_", str(value).strip().lower()).strip("_")


def derive_record(raw):
    """Re-derive every semantic field from the raw message text."""
    out = {
        "parser_status": "ok",
        "raw_type": None,
        "level_derived": None,
        "msg_ts_ms": None,
        "service_msg": None,
        "node_ip": None,
        "start_ms": None,
        "end_ms": None,
        "duration_seconds": None,
        "duration_how": None,
        "start_semantics": None,
        "semantic_role": None,
        "gt_candidate": False,
        "gt_included": False,
        "decision_reason": None,
        "pattern": None,
    }
    message = raw["message"]
    parts = message.split(" | ", 5)
    if len(parts) < 6:
        fragment = message.strip().startswith("(Background on this error at")
        kind = "traceback_continuation" if fragment else "message_format_error"
        out["parser_status"] = kind
        out["raw_type"] = kind
        out["level_derived"] = ""
        out["service_msg"] = ""
        out["semantic_role"] = "parse_error"
        out["decision_reason"] = (
            "multi-line traceback continuation fragment of an operational ERROR "
            "log-upload record; not an independent event (parent-row adjacency and "
            "service match verified)"
            if fragment
            else "message does not have 6 '|'-separated raw segments"
        )
        out["pattern"] = normalize_pattern(message)[:160]
        return out
    out["msg_ts_ms"] = parse_msg_ts(parts[0])
    out["level_derived"] = parts[1].strip()
    out["node_ip"] = parts[2].strip()
    out["service_msg"] = parts[4].strip()
    payload = parts[5]
    out["pattern"] = normalize_pattern(payload)
    if out["msg_ts_ms"] is None:
        out["parser_status"] = "message_ts_unparseable"

    raw_type = classify(payload, out["level_derived"])
    out["raw_type"] = raw_type

    if raw_type in START_SEMANTICS:
        semantics = START_SEMANTICS[raw_type]
        out["start_semantics"] = semantics
        duration, how = parse_duration_seconds(payload)
        if duration is None and raw_type in DEFAULT_DURATION:
            duration, how = DEFAULT_DURATION[raw_type], "type_default"
        out["duration_seconds"] = duration
        out["duration_how"] = how
        if semantics == "embedded":
            start = parse_embedded_ts(payload)
            if start is None:
                out["parser_status"] = "embedded_start_missing"
            out["start_ms"] = start
        else:
            out["start_ms"] = out["msg_ts_ms"]
        if out["start_ms"] is not None and duration is not None:
            out["end_ms"] = out["start_ms"] + int(round(duration * 1000))
        # roles / GT decisions
        if raw_type in SUPPORTED_FAULT_TYPES:
            out["semantic_role"] = "fault_injection"
            out["gt_candidate"] = True
            if out["parser_status"] == "ok" and out["start_ms"] is not None and out["end_ms"] is not None:
                out["gt_included"] = True
                out["decision_reason"] = "explicit fault-injection record; interval re-derived from raw message"
            else:
                out["gt_included"] = False
                out["decision_reason"] = "supported type but parse incomplete: " + out["parser_status"]
        elif raw_type == "normal_memory_freed":
            out["semantic_role"] = "possible_fault"
            out["gt_candidate"] = True
            out["gt_included"] = False
            out["decision_reason"] = (
                "V2 s6 candidate independent fault (free_using_memory); excluded from the "
                "primary caliber pending final decision; see normal_memory_freed_audit.json"
            )
    elif raw_type == "log_upload_failure":
        out["semantic_role"] = "generic_error_system_record"
        out["start_ms"] = out["msg_ts_ms"]
        out["end_ms"] = out["msg_ts_ms"]
        out["duration_seconds"] = 0.0
        out["duration_how"] = "point_event"
        out["decision_reason"] = (
            "operational log-upload failure record; no injection wording, no duration, "
            "no start-at field (V2 s7: semantics_not_established as fault)"
        )
    elif raw_type == "exception_injection_failed":
        out["semantic_role"] = "unsupported"
        out["start_ms"] = out["msg_ts_ms"]
        out["end_ms"] = out["msg_ts_ms"]
        out["duration_seconds"] = 0.0
        out["duration_how"] = "point_event"
        out["decision_reason"] = (
            "injection attempt reported an internal error; no fault interval established"
        )
    elif raw_type in ("normal", "upload_success"):
        out["semantic_role"] = "normal_system_record"
        out["decision_reason"] = "INFO-level routine system record (V2 s8: never failure GT)"
    else:
        out["start_ms"] = out["msg_ts_ms"]
        if out["level_derived"] == "INFO":
            out["semantic_role"] = "normal_system_record"
            out["decision_reason"] = (
                "INFO-level operational record without fault semantics; pattern "
                "preserved in registry and inventory (V2 s8)"
            )
        else:
            out["semantic_role"] = "unsupported"
            out["decision_reason"] = (
                "unclassified non-INFO pattern; preserved in registry, "
                "not silently dropped (V2 s8)"
            )
        if out["parser_status"] == "ok":
            out["parser_status"] = "unclassified_pattern"
    return out


def payload_of(r):
    parts = r["raw"]["message"].split(" | ", 5)
    return parts[5] if len(parts) >= 6 else r["raw"]["message"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    with open(RUN_TABLE, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for idx, raw in enumerate(reader):
            rec = derive_record(raw)
            rec["source_index"] = idx
            rec["raw"] = raw
            rows.append(rec)

    total = len(rows)
    type_counts = Counter(r["raw_type"] for r in rows)
    role_counts = Counter(r["semantic_role"] for r in rows)
    level_counts = Counter(r["level_derived"] for r in rows)
    pattern_counts = Counter((r["level_derived"], r["raw_type"], r["pattern"]) for r in rows)

    # ---------------- per-type timing semantics evidence ----------------
    semantics_table = {}
    for t in sorted(type_counts):
        subset = [r for r in rows if r["raw_type"] == t]
        offs = [
            (r["msg_ts_ms"] - r["start_ms"]) / 1000.0
            for r in subset
            if r["msg_ts_ms"] is not None and r["start_ms"] is not None
            and r["start_semantics"] == "embedded"
        ]
        durs = Counter(r["duration_how"] for r in subset)
        dur_values = sorted(
            Counter(round(r["duration_seconds"], 6) for r in subset if r["duration_seconds"] is not None).items()
        )
        entry = {
            "count": len(subset),
            "start_semantics": START_SEMANTICS.get(t, "message_ts(point/none)"),
            "duration_how": dict(durs),
            "distinct_duration_values_seconds": [{"value": v, "count": c} for v, c in dur_values[:40]],
            "sample_payloads": [payload_of(r)[:220] for r in subset[:3]],
        }
        if offs:
            offs_sorted = sorted(offs)
            entry["msg_ts_minus_start_seconds"] = {
                "min": offs_sorted[0],
                "median": offs_sorted[len(offs_sorted) // 2],
                "max": offs_sorted[-1],
                "interpretation": (
                    "message logged at END of fault window (offset ~ duration)"
                    if offs_sorted[len(offs_sorted) // 2] > 30
                    else "message logged at START of fault window (offset ~ 0)"
                ),
            }
        semantics_table[t] = entry

    # ---------------- normal memory freed audit ----------------
    freed = [r for r in rows if r["raw_type"] == "normal_memory_freed"]
    mem_by_service = defaultdict(list)
    for r in rows:
        if r["raw_type"] == "memory_anomalies" and r["start_ms"] is not None:
            mem_by_service[r["service_msg"]].append(r)
    for v in mem_by_service.values():
        v.sort(key=lambda r: r["start_ms"])
    freed_audit = []
    for r in freed:
        prev = [m for m in mem_by_service.get(r["service_msg"], []) if m["start_ms"] < r["msg_ts_ms"]]
        nearest = prev[-1] if prev else None
        entry = {
            "source_index": r["source_index"],
            "service": r["service_msg"],
            "message_ts_iso": iso_ms(r["msg_ts_ms"]),
            "payload": payload_of(r),
            "preparsed_anomaly_type": r["raw"]["anomaly_type"],
            "preparsed_duration_seconds": r["raw"]["duration_seconds"],
            "derived_duration_seconds": r["duration_seconds"],
            "derived_duration_how": r["duration_how"],
            "nearest_preceding_memory_anomaly": None,
        }
        if nearest is not None:
            entry["nearest_preceding_memory_anomaly"] = {
                "source_index": nearest["source_index"],
                "start_iso": iso_ms(nearest["start_ms"]),
                "end_iso": iso_ms(nearest["end_ms"]),
                "gap_msg_ts_minus_prev_end_seconds": (r["msg_ts_ms"] - nearest["end_ms"]) / 1000.0,
                "gap_msg_ts_minus_prev_start_seconds": (r["msg_ts_ms"] - nearest["start_ms"]) / 1000.0,
            }
        freed_audit.append(entry)
    gaps_end = [
        e["nearest_preceding_memory_anomaly"]["gap_msg_ts_minus_prev_end_seconds"]
        for e in freed_audit if e["nearest_preceding_memory_anomaly"]
    ]
    freed_summary = {
        "count": len(freed),
        "all_payloads_identical_pattern": len(set(e["payload"] for e in freed_audit)) == 1,
        "gap_to_prev_memory_end_seconds": {
            "min": min(gaps_end) if gaps_end else None,
            "median": sorted(gaps_end)[len(gaps_end) // 2] if gaps_end else None,
            "max": max(gaps_end) if gaps_end else None,
            "count_with_preceding_memory_event": len(gaps_end),
        },
        "end_marker_hypothesis": (
            "REJECTED: message ts is far from the preceding injection end; records look "
            "independent (see per-record gaps)"
            if gaps_end and min(gaps_end) > 60
            else "NOT REJECTED: at least one record sits at a preceding injection end"
            if gaps_end
            else "UNTESTABLE: no preceding memory event"
        ),
        "records": freed_audit,
        "decision": (
            "primary caliber: EXCLUDED (possible_fault, pending verification); "
            "alternate caliber: INCLUDED as free_using_memory with start=message ts, "
            "duration=600s (matches Ada-MGAD main pre_GAIA.py:190-196 which labels these "
            "records positive, and DiagFusion gaia_resplit.csv which keeps 16/17 as an "
            "independent fault class with st_time=message ts, duration=600)"
        ),
    }

    # ---------------- ERROR audit ----------------
    err_rows = [r for r in rows if r["level_derived"] == "ERROR"]
    err_audit = {
        "count": len(err_rows),
        "by_raw_type": dict(Counter(r["raw_type"] for r in err_rows)),
        "records": [
            {
                "source_index": r["source_index"],
                "service": r["service_msg"],
                "message_ts_iso": iso_ms(r["msg_ts_ms"]),
                "payload": payload_of(r)[:300],
                "raw_type": r["raw_type"],
                "has_injection_wording": bool(
                    TS_EMBEDDED_RE.search(r["raw"]["message"]) or r["duration_how"] not in (None, "point_event")
                ),
                "has_duration": r["duration_how"] not in (None, "point_event"),
                "recommended_gt_included": False,
                "reason": r["decision_reason"],
            }
            for r in err_rows
        ],
        "level_column_mismatch_rows": [
            r["source_index"] for r in rows
            if r["level_derived"] and r["level_derived"] != str(r["raw"]["level"]).strip()
        ],
    }

    # ---------------- CPU audit ----------------
    cpu = [r for r in rows if r["raw_type"] == "cpu_anomalies"]
    cpu_audit = {
        "count": len(cpu),
        "float_regex": r"lasts\s+(\d+(?:\.\d+)?)\s+seconds",
        "records": [
            {
                "source_index": r["source_index"],
                "service": r["service_msg"],
                "payload": payload_of(r)[:300],
                "embedded_start_iso": iso_ms(r["start_ms"]),
                "derived_duration_seconds": r["duration_seconds"],
                "duration_how": r["duration_how"],
                "preparsed_duration_seconds": r["raw"]["duration_seconds"],
                "duration_matches_preparse": (
                    r["duration_seconds"] is not None
                    and abs(r["duration_seconds"] - float(r["raw"]["duration_seconds"] or "nan")) < 1e-3
                ),
                "all_digits_preserved": bool(
                    re.search(r"lasts\s+\d+\.\d+\s+seconds", r["raw"]["message"])
                ) == (r["duration_seconds"] is not None and float(r["duration_seconds"]) % 1 != 0),
            }
            for r in cpu
        ],
        "parse_failures": [r["source_index"] for r in cpu if r["duration_seconds"] is None],
    }

    # ---------------- pre-parse cross-check ----------------
    mismatches = []
    per_field = Counter()
    type_pair_counts = Counter()
    non_fault_unlabeled = {
        "upload_success", "unknown", "normal",
        "message_format_error", "traceback_continuation",
    }
    not_comparable = {"message_format_error", "traceback_continuation"}
    skipped = 0
    for r in rows:
        raw = r["raw"]
        derived = r
        if derived["parser_status"] in not_comparable:
            skipped += 1
            continue
        fields = {}
        fields["level"] = (
            str(raw["level"]).strip() == (derived["level_derived"] or "")
        )
        # timestamp column is second-resolution message ts, space-separated
        fields["timestamp"] = (
            derived["msg_ts_ms"] is not None
            and str(raw["timestamp"]).strip()
            == iso_ms(derived["msg_ts_ms"])[:19].replace("T", " ")
        )
        fields["service"] = str(raw["service"]).strip() == (derived["service_msg"] or "")
        fields["service_node"] = str(raw["service_node"]).strip() == (derived["service_msg"] or "")
        pre_type = str(raw["anomaly_type"]).strip()
        fields["anomaly_type"] = (
            norm_type_token(pre_type) == norm_type_token(derived["raw_type"])
            or (pre_type == "" and derived["raw_type"] in non_fault_unlabeled)
        )
        if not fields["anomaly_type"]:
            type_pair_counts[(pre_type, str(derived["raw_type"]))] += 1
        pre_dur = str(raw["duration_seconds"]).strip()
        if derived["duration_seconds"] is None:
            fields["duration_seconds"] = pre_dur in ("", "nan", "None")
        elif pre_dur in ("", "nan", "None"):
            # owner leaves duration blank for non-injection point records
            fields["duration_seconds"] = derived["duration_seconds"] == 0.0
        else:
            try:
                fields["duration_seconds"] = abs(float(pre_dur) - derived["duration_seconds"]) < 1e-3
            except ValueError:
                fields["duration_seconds"] = False
        bad = [k for k, ok in fields.items() if not ok]
        for k in bad:
            per_field[k] += 1
        if bad:
            mismatches.append(
                {
                    "source_index": r["source_index"],
                    "fields": bad,
                    "preparsed": {k: raw[k] for k in ("timestamp", "level", "service_node", "anomaly_type", "duration_seconds")},
                    "derived": {
                        "msg_ts_iso": iso_ms(derived["msg_ts_ms"]),
                        "level": derived["level_derived"],
                        "service_msg": derived["service_msg"],
                        "raw_type": derived["raw_type"],
                        "duration_seconds": derived["duration_seconds"],
                    },
                }
            )
    crosscheck = {
        "total_rows": total,
        "rows_not_comparable": skipped,
        "rows_not_comparable_reason": "traceback continuation fragments / malformed rows; all preparsed columns blank",
        "rows_with_any_mismatch": len(mismatches),
        "mismatch_count_by_field": dict(per_field),
        "anomaly_type_label_pairs_preparse_vs_derived": [
            {"preparsed": a, "derived": b, "count": c} for (a, b), c in type_pair_counts.most_common()
        ],
        "examples": mismatches[:20],
    }
    if len(mismatches) > 100:
        with open(os.path.join(OUT_DIR, "preparse_mismatches.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source_index", "fields", "preparsed", "derived"])
            for m in mismatches:
                writer.writerow([m["source_index"], ";".join(m["fields"]), json.dumps(m["preparsed"]), json.dumps(m["derived"])])
        crosscheck["full_detail_csv"] = "preparse_mismatches.csv"

    # ---------------- GT registry + distributions ----------------
    gt_primary = [r for r in rows if r["gt_included"]]
    gt_with_freed = gt_primary + [r for r in freed if r["start_ms"] is not None and r["end_ms"] is not None]

    def distributions(events):
        by_type = Counter(r["raw_type"] for r in events)
        by_service = Counter(r["service_msg"] for r in events)
        by_type_service = Counter((r["raw_type"], r["service_msg"]) for r in events)
        starts = [r["start_ms"] for r in events]
        ends = [r["end_ms"] for r in events]
        return {
            "n_gt": len(events),
            "by_fault_type": dict(sorted(by_type.items())),
            "by_labelled_service": dict(sorted(by_service.items())),
            "by_fault_type_and_service": [
                {"fault_type": a, "service": b, "count": c} for (a, b), c in sorted(by_type_service.items())
            ],
            "start_ms_min": min(starts) if starts else None,
            "start_ms_max": max(starts) if starts else None,
            "end_ms_max": max(ends) if ends else None,
            "start_iso_min": iso_ms(min(starts)) if starts else None,
            "start_iso_max": iso_ms(max(starts)) if starts else None,
        }

    # same-bin collision statistics (30s bins keyed by event start)
    def collision_stats(events):
        bin_all = Counter(r["start_ms"] // BIN_MS for r in events)
        bin_svc = Counter((r["service_msg"], r["start_ms"] // BIN_MS) for r in events)
        multi = {k: c for k, c in bin_all.items() if c > 1}
        multi_svc = {k: c for k, c in bin_svc.items() if c > 1}
        services_per_bin = defaultdict(set)
        for r in events:
            services_per_bin[r["start_ms"] // BIN_MS].add(r["service_msg"])
        multi_service_bins = {k: len(v) for k, v in services_per_bin.items() if len(v) > 1}
        return {
            "bins_with_multiple_injections": len(multi),
            "max_injections_in_one_bin": max(multi.values()) if multi else 0,
            "same_service_same_bin_bins": len(multi_svc),
            "max_same_service_same_bin": max(multi_svc.values()) if multi_svc else 0,
            "bins_with_multiple_services": len(multi_service_bins),
            "max_services_in_one_bin": max(multi_service_bins.values()) if multi_service_bins else 0,
        }

    # events whose start precedes the metric-based detector timeline
    pre_timeline = [r for r in gt_primary if r["start_ms"] < METRIC_TIMELINE_START_MS]
    pre_timeline_freed = [r for r in gt_with_freed if r["start_ms"] < METRIC_TIMELINE_START_MS]

    excluded = [r for r in rows if not r["gt_included"]]
    exclusion_breakdown = Counter(
        (r["raw_type"], r["semantic_role"]) for r in excluded
    )

    summary = {
        "taxonomy_version": TAXONOMY_VERSION,
        "run_table": RUN_TABLE,
        "raw_record_count": total,
        "level_distribution_derived": dict(sorted(level_counts.items())),
        "raw_type_distribution": dict(sorted(type_counts.items())),
        "semantic_role_distribution": dict(sorted(role_counts.items())),
        "message_pattern_inventory": [
            {"level": lv, "raw_type": rt, "pattern": pat, "count": c}
            for (lv, rt, pat), c in pattern_counts.most_common()
        ],
        "per_type_semantics": semantics_table,
        "primary_caliber": distributions(gt_primary),
        "with_freed_caliber": distributions(gt_with_freed),
        "exclusion_breakdown": [
            {"raw_type": a, "semantic_role": b, "count": c}
            for (a, b), c in sorted(exclusion_breakdown.items())
        ],
        "same_bin_collisions_primary": collision_stats(gt_primary),
        "same_bin_collisions_with_freed": collision_stats(gt_with_freed),
        "events_starting_before_metric_timeline": {
            "metric_timeline_start_ms": METRIC_TIMELINE_START_MS,
            "metric_timeline_start_iso": iso_ms(METRIC_TIMELINE_START_MS),
            "primary_count": len(pre_timeline),
            "primary_by_type": dict(Counter(r["raw_type"] for r in pre_timeline)),
            "with_freed_count": len(pre_timeline_freed),
            "note": "these events precede the first metric sample (2021-07-01 18:00 CST); logs/traces exist from 09:57 CST",
        },
        "hardcoded_expectations": "none; every count in this file is an output of parsing",
    }

    # ---------------- write artifacts ----------------
    reg_path = os.path.join(OUT_DIR, "run_record_registry.csv")
    with open(reg_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "source_index", "datetime", "service", "raw_message", "level", "raw_type",
            "parsed_start_iso", "parsed_start_ms", "parsed_end_ms", "parsed_duration_seconds",
            "duration_source", "start_semantics", "parser_status", "semantic_role",
            "gt_candidate", "gt_included", "decision_reason",
        ])
        for r in rows:
            writer.writerow([
                r["source_index"], r["raw"]["datetime"], r["raw"]["service"], r["raw"]["message"],
                r["level_derived"], r["raw_type"],
                iso_ms(r["start_ms"]), r["start_ms"] if r["start_ms"] is not None else "",
                r["end_ms"] if r["end_ms"] is not None else "",
                r["duration_seconds"] if r["duration_seconds"] is not None else "",
                r["duration_how"] or "", r["start_semantics"] or "", r["parser_status"],
                r["semantic_role"], r["gt_candidate"], r["gt_included"], r["decision_reason"],
            ])

    gt_path = os.path.join(OUT_DIR, "gt_event_registry.csv")
    with open(gt_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "case_id", "source_index", "raw_type", "fault_type", "labelled_service",
            "start_ms", "end_ms", "duration_seconds", "taxonomy_version", "evidence_status",
        ])
        for r in gt_with_freed:
            status = (
                "candidate_pending_verification"
                if r["raw_type"] == "normal_memory_freed"
                else "verified_from_raw_message"
            )
            writer.writerow([
                "gaia-v2-%05d" % r["source_index"], r["source_index"], r["raw_type"], r["raw_type"],
                r["service_msg"], r["start_ms"], r["end_ms"],
                round(r["duration_seconds"], 6), TAXONOMY_VERSION, status,
            ])

    def dump(name, obj):
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)

    dump("taxonomy_summary.json", summary)
    dump("normal_memory_freed_audit.json", freed_summary)
    dump("error_records_audit.json", err_audit)
    dump("cpu_anomaly_audit.json", cpu_audit)
    dump("preparse_crosscheck.json", crosscheck)

    # ---------------- stdout summary ----------------
    print("total rows:", total)
    print("raw_type_distribution:", json.dumps(dict(sorted(type_counts.items())), ensure_ascii=False))
    print("role_distribution:", json.dumps(dict(sorted(role_counts.items())), ensure_ascii=False))
    print("level_distribution:", json.dumps(dict(sorted(level_counts.items())), ensure_ascii=False))
    print("N_GT primary:", len(gt_primary), " with freed:", len(gt_with_freed))
    print("primary by type:", json.dumps(summary["primary_caliber"]["by_fault_type"]))
    print("primary by service:", json.dumps(summary["primary_caliber"]["by_labelled_service"]))
    print("primary start range:", summary["primary_caliber"]["start_iso_min"], "->", summary["primary_caliber"]["start_iso_max"])
    print("freed gaps(msg-prev_mem_end)s:", freed_summary["gap_to_prev_memory_end_seconds"], freed_summary["end_marker_hypothesis"])
    print("ERROR rows:", err_audit["count"], err_audit["by_raw_type"], "level-col mismatches:", len(err_audit["level_column_mismatch_rows"]))
    print("CPU rows:", cpu_audit["count"], "parse failures:", cpu_audit["parse_failures"])
    print("crosscheck mismatch rows:", crosscheck["rows_with_any_mismatch"], crosscheck["mismatch_count_by_field"])
    print("type pair mismatches:", crosscheck["anomaly_type_label_pairs_preparse_vs_derived"][:10])
    print("collisions primary:", summary["same_bin_collisions_primary"])
    print("pre-metric-timeline events:", summary["events_starting_before_metric_timeline"]["primary_count"],
          summary["events_starting_before_metric_timeline"]["primary_by_type"])
    unk = [r for r in rows if r["raw_type"] == "unknown"]
    print("unknown rows:", len(unk))
    for r in unk[:10]:
        print("  UNKNOWN sample idx", r["source_index"], ":", r["raw"]["message"][:200])
    print("artifacts written to", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
