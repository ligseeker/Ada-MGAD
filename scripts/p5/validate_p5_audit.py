#!/usr/bin/env python3
"""Fail-closed consistency checks for the P5-G0R2 evidence package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p5.common import sha256_file, write_json  # noqa: E402


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/p5/g0r2")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs/P5_GAIA_RECONCILIATION_AUDIT_V1.md")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.artifact_dir / "validation_results.json"
    required = {
        name: args.artifact_dir / name
        for name in (
            "event_revalidation.json",
            "event_registry.csv",
            "label_reconciliation.json",
            "label_reconciliation.csv",
            "temporal_resolution.json",
            "raw_telemetry_timing.json",
            "rca_compatibility.json",
            "split_feasibility.json",
            "historical_reuse_matrix.csv",
            "evidence_ledger.csv",
            "repository_state.json",
        )
    }
    checks = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    check("required_artifacts_exist", all(path.is_file() for path in required.values()), {
        name: path.is_file() for name, path in required.items()
    })
    if not all(path.is_file() for path in required.values()):
        write_json(output, {"status": "FAIL", "checks": checks})
        raise SystemExit("required artifact missing")

    event = _load(required["event_revalidation.json"])
    label = _load(required["label_reconciliation.json"])
    temporal = _load(required["temporal_resolution.json"])
    timing = _load(required["raw_telemetry_timing.json"])
    compatibility = _load(required["rca_compatibility.json"])
    split = _load(required["split_feasibility.json"])
    with required["event_registry.csv"].open(encoding="utf-8", newline="") as handle:
        registry_count = sum(1 for _ in csv.DictReader(handle))
    registry_sha = sha256_file(required["event_registry.csv"])

    check("event_inventory", event["raw_record_count"] == 17152 and event["supported_injections"] == 16200 and event["excluded_records"] == 952, {
        "raw": event["raw_record_count"], "supported": event["supported_injections"], "excluded": event["excluded_records"]
    })
    check("registry_count", registry_count == 16200, registry_count)
    bound_hashes = {
        "temporal": temporal["source"]["event_registry_sha256"],
        "compatibility": compatibility["source"]["event_registry_sha256"],
        "split": split["event_registry"]["sha256"],
    }
    check("registry_hash_binding", all(value == registry_sha for value in bound_hashes.values()), {
        "actual": registry_sha, **bound_hashes
    })
    check("label_coupling", label["coupling"]["current_ad_mappable_supported_injections"] == 16188 and label["coupling"]["service_exact_matches_among_current_ad_mappable"] == 16188, label["coupling"])
    check("cpu_defect", event["cpu_parser_audit"]["status"] == "REVALIDATED PARSER DEFECT" and event["cpu_parser_audit"]["current_main_parse_failure"] == 12, event["cpu_parser_audit"])
    scan = timing["traces"]["targeted_full_status_and_duration_scan"]
    check("trace_status_full_count", sum(scan["status_code_distribution"].values()) == scan["rows"] == 28681438, scan["status_code_distribution"])
    check("trace_rules_not_equivalent", scan["rules_equivalent"] is False and scan["rows_missed_by_ge_500"] == 3071740, {"rules_equivalent": scan["rules_equivalent"], "missed": scan["rows_missed_by_ge_500"]})
    check("context_group_consistency", compatibility["context_overlap"]["300"]["injection_union_context"]["groups"] == split["protocol_s_component_integration"]["windows"]["300"]["group_count"] == 322 and compatibility["context_overlap"]["600"]["injection_union_context"]["groups"] == split["protocol_s_component_integration"]["windows"]["600"]["group_count"] == 23, "W300=322; W600=23")
    check("no_exact_68d_claim", all(not row["exact_68d_representation_built"] for row in compatibility["configuration_health"].values()), compatibility["configuration_health"])
    check("split_protocols_distinct", split["protocol_s_component_integration"]["windows"]["300"]["feasible"] is True and split["protocol_s_component_integration"]["windows"]["600"]["feasible"] is False and len(split["protocol_t_continuous_temporal_e2e"]["protocols"]) >= 2, "Protocol S W300 feasible; W600 infeasible; two Protocol T candidates")

    with required["evidence_ledger.csv"].open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    mandatory = {"claim_id", "claim", "status", "repository", "branch", "file_or_dataset", "location", "evidence", "confidence", "reproduction_command", "artifact"}
    check("evidence_ledger_schema", len(ledger) >= 20 and mandatory.issubset(ledger[0]) and len({row["claim_id"] for row in ledger}) == len(ledger), {"rows": len(ledger), "fields": sorted(ledger[0])})
    with required["historical_reuse_matrix.csv"].open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    check("historical_reuse_matrix", len(matrix) >= 7, {"rows": len(matrix), "assets": [row["historical_asset"] for row in matrix]})

    report_text = args.report.read_text(encoding="utf-8")
    required_sections = ["## {}.".format(index) for index in range(1, 18)] + ["## Q1–Q20 Answers", "# P5 Decision Table", "本轮没有训练正式 Ada-MGAD / Ada-RCA"]
    check("report_structure", all(value in report_text for value in required_sections), {"missing": [value for value in required_sections if value not in report_text]})
    production_diff = subprocess.check_output(
        ("git", "-C", str(PROJECT_ROOT), "diff", "--name-only", "HEAD", "--", "main.py", "src", "util"),
        text=True,
    ).splitlines()
    check("production_code_unchanged", not production_diff, production_diff)

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    report = {
        "schema_version": "p5_g0r2_validation_v1",
        "status": status,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope": "artifact consistency only; no model training or performance evaluation",
    }
    write_json(output, report)
    print("{}: {}/{} checks passed".format(status, report["checks_passed"], report["checks_total"]))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
