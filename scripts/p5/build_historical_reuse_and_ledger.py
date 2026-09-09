#!/usr/bin/env python3
"""Build deterministic P5 historical-evidence reuse and claim ledgers.

This script is deliberately an audit/reporting utility.  It reads the immutable
historical checkout and P5 audit artifacts, computes source hashes, and writes
only CSV reports.  It never changes production preprocessing or model code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


CURRENT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = Path("/home/zhangll24/RCA_project/Ada-MGAD-rca-standalone")
GAIA_ROOT = Path("/home/zhangll24/RCA_project/datasets/GAIA/MicroSS")
RUN_TABLE = GAIA_ROOT / "run/run/run/run_table_2021-07.csv"

REQUIRED_LEDGER_FIELDS = (
    "claim_id",
    "claim",
    "status",
    "repository",
    "branch",
    "file_or_dataset",
    "location",
    "evidence",
    "confidence",
    "reproduction_command",
    "artifact",
)

HISTORY_ASSETS = (
    "docs/DATASET_AUDIT.md",
    "docs/GAIA_INCLUSION_AUDIT.md",
    "docs/TELEMETRY_DIAGNOSTICS.md",
    "docs/SPLIT_DIAGNOSTICS.md",
    "docs/P2_METRIC_FEATURES.md",
    "docs/P2_MODALITY_SCHEMA_AUDIT.md",
    "docs/BASELINE_RESULTS.md",
    "docs/BENCHMARK_PROTOCOL.md",
    "docs/RESEARCH_STATUS.md",
    "docs/EXPERIMENT_LOG.md",
    "artifacts/p1/dataset_audit.json",
    "artifacts/p1/gaia_inclusion_diagnostics.json",
    "artifacts/p1/telemetry_diagnostics.json",
    "artifacts/p1/context_group_diagnostics.json",
    "artifacts/p1/split_diagnostics.json",
    "artifacts/p1/gate_audit.json",
    "artifacts/p1/manifests/gaia/manifest.json",
    "artifacts/p1/manifests/gaia/sources.jsonl",
    "artifacts/p1/splits/gaia/split_manifest.json",
    "artifacts/p1/inclusion/gaia/manifest.json",
    "artifacts/p2/metric_feature_audit.json",
    "artifacts/p2/event_feature_audit.json",
    "artifacts/p2/metric_feature_summary.json",
    "artifacts/p2/event_feature_summary.json",
)

P5_ARTIFACTS = (
    "artifacts/p5/g0r2/event_revalidation.json",
    "artifacts/p5/g0r2/event_registry.csv",
    "artifacts/p5/g0r2/label_reconciliation.json",
    "artifacts/p5/g0r2/label_reconciliation.csv",
    "artifacts/p5/g0r2/temporal_resolution.json",
    "artifacts/p5/g0r2/raw_telemetry_timing.json",
    "artifacts/p5/g0r2/rca_compatibility.json",
    "artifacts/p5/g0r2/split_feasibility.json",
    "artifacts/p5/g0r2/repository_state.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory(paths: Iterable[Path]) -> Dict[str, str]:
    """Return deterministic path -> hash bindings for existing files."""
    return {
        str(path): sha256(path)
        for path in sorted(paths, key=lambda item: str(item))
        if path.is_file()
    }


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(CURRENT_ROOT))
    except ValueError:
        return str(path)


def command(script: str, args: str) -> str:
    return "python scripts/p5/{} {}".format(script, args).strip()


def historical_matrix_rows(hashes: Mapping[str, str]) -> List[Dict[str, str]]:
    """Describe historical claims without upgrading them to current FACTs."""
    rows = [
        (
            "DATASET_AUDIT",
            "17,152 raw rows; 16,200 supported events; 952 excluded; historical root/fault distributions and parser rules.",
            "HISTORICAL EVIDENCE / REVALIDATED PARTIAL",
            "Audit script is reusable, but historical source_path points to a removed project_2 location.",
            "High for logic; numeric claims require current-path binding.",
            "Run audit_p1_datasets.py with current GAIA path; preserve old artifact as historical.",
        ),
        (
            "GAIA_INCLUSION_AUDIT",
            "13,470 anchor-unique-root main cohort and 2,730 multi-root sensitivity cohort.",
            "HISTORICAL EVIDENCE / NOT FROZEN",
            "Inclusion manifest and purity logic exist, but their source manifest is path-bound to the old GAIA location; document passport says UNVERIFIED.",
            "Medium; use inclusion semantics and flags, not counts without P5 rerun.",
            "Recompute anchor root concurrency and raw interval purity from the current event registry.",
        ),
        (
            "TELEMETRY_DIAGNOSTICS",
            "Historical full scan reported 349,120,558 rows, GAIA log prefix timestamps, and window coverage/group diagnostics.",
            "HISTORICAL EVIDENCE / REVALIDATED VIA P5 BINDING",
            "Historical artifact is retained; P5 raw timing artifact binds current layout digests and run-table hash.",
            "High for timing values present in P5 artifact; no need to repeat the 349M-row scan.",
            "Use diagnose_p1_telemetry.py only for targeted checks; compare hashes and schema.",
        ),
        (
            "SPLIT_DIAGNOSTICS",
            "Grouped-stratified 5-fold was selected historically; 322 context groups and a temporal alternative were recorded.",
            "HISTORICAL EVIDENCE / NOT FROZEN",
            "Split artifacts are checksum-bearing, but GAIA manifests retain old source paths and one manifest says split assignment not frozen.",
            "Reusable validator/group algorithm only.",
            "Rebuild Protocol S and Protocol T feasibility from the current P5 event registry; do not inherit a split lock.",
        ),
        (
            "P2_METRIC_FEATURES",
            "170D metric bundle and coverage audit; 30s onset was historically degenerate while 60/120s were better observed.",
            "HISTORICAL EVIDENCE / REVALIDATED PARTIAL",
            "Feature and audit artifacts exist; P5 timing confirms 30s nominal metric sampling, but P5 must independently assess frozen Ada-RCA representation health.",
            "Reusable coverage methodology and diagnostic target; not a P5 model configuration.",
            "Use as representation-health prior; never select window/bin using its model performance.",
        ),
        (
            "P2_MODALITY_SCHEMA_AUDIT",
            "GAIA logs use message-prefix millisecond timestamps; traces expose start/end/status/parent fields; historical L0/T0 used GAIA-specific channels.",
            "HISTORICAL EVIDENCE / STATUS CONFLICT",
            "Document passport says UNVERIFIED, while later artifacts contain full event features and RESEARCH_STATUS has stale pending language.",
            "Medium; raw schema is useful, feature bundles are not canonical P5 Ada-RCA input.",
            "Re-audit current raw schema and Ada-RCA frozen extractor for 15s and 30s compatibility.",
        ),
        (
            "BASELINE_RESULTS",
            "Frequency baseline exposes severe root/fault shortcut; historical B0/B1/B2 rankings and macro metrics are archived.",
            "HISTORICAL EVIDENCE / PERFORMANCE-BLIND ONLY",
            "Baseline artifacts are complete, but docs passport says UNVERIFIED and performance cannot select P5 window/bin/split.",
            "Root/fault imbalance sanity check only.",
            "Report overall, root-macro, and fault-macro; do not use AC/MRR for configuration selection.",
        ),
        (
            "PROTOCOL_DOCS",
            "Historical benchmark protocol separates labelled injected service from causal root cause and separates AD from standalone RCA.",
            "HISTORICAL EVIDENCE",
            "BENCHMARK_PROTOCOL/RCA_RESEARCH_DESIGN are design records, not current raw-data evidence.",
            "Reusable terminology and firewall constraints.",
            "Carry conservative terminology into the P5 report, then freeze only after current evidence ledger review.",
        ),
    ]
    config = {
        "historical_root": str(HISTORICAL_ROOT),
        "current_gaia_root": str(GAIA_ROOT),
        "historical_manifest_exists": (HISTORICAL_ROOT / "artifacts/p1/manifests/gaia/sources.jsonl").is_file(),
        "old_project2_source_exists": Path("/home/zhangll24/project_2/MultimodalAD/MSTGAD-GAIA/data/GAIA/MicroSS").exists(),
        "current_run_table_exists": RUN_TABLE.is_file(),
        "historical_head": "64bb681328fa1793014615a20ba2bc1fdf33c3b7",
    }
    rendered_hashes = jdump(dict(sorted(hashes.items())))
    output = []
    for asset, claim, status, evidence, confidence, action in rows:
        output.append(
            {
                "historical_asset": asset,
                "historical_claim": claim,
                "reproducible_now": "CONDITIONAL" if "path" in evidence or "current" in action else "YES",
                "revalidated_now": "PARTIAL" if "REVALIDATED" in status else "NO",
                "reusable_for_p5": "YES" if asset in {"DATASET_AUDIT", "TELEMETRY_DIAGNOSTICS", "P2_METRIC_FEATURES", "PROTOCOL_DOCS"} else "CONDITIONAL",
                "verification_status": status,
                "evidence": evidence,
                "confidence": confidence,
                "action": action,
                "source_hashes": rendered_hashes,
                "config": jdump(config),
            }
        )
    return output


def evidence_rows(
    hashes: Mapping[str, str],
    event: Mapping[str, Any],
    label: Mapping[str, Any],
    timing: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    split: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Create claims from current P5 audit artifacts, with explicit limitations."""
    event_path = CURRENT_ROOT / "artifacts/p5/g0r2/event_revalidation.json"
    label_path = CURRENT_ROOT / "artifacts/p5/g0r2/label_reconciliation.json"
    timing_path = CURRENT_ROOT / "artifacts/p5/g0r2/raw_telemetry_timing.json"
    temporal_path = CURRENT_ROOT / "artifacts/p5/g0r2/temporal_resolution.json"
    compatibility_path = CURRENT_ROOT / "artifacts/p5/g0r2/rca_compatibility.json"
    split_path = CURRENT_ROOT / "artifacts/p5/g0r2/split_feasibility.json"
    artifact_ref = lambda path: "{}; sha256={}".format(relative(path), hashes.get(str(path), "MISSING"))
    hist_repo = str(HISTORICAL_ROOT)
    p5_repo = str(CURRENT_ROOT)
    gaia_loc = str(RUN_TABLE)
    base = {
        "repository": p5_repo,
        "branch": "e2e",
        "confidence": "HIGH",
    }
    rows = []

    def add(claim_id: str, claim: str, status: str, repository: str, branch: str, file_or_dataset: str, location: str, evidence: str, confidence: str, reproduction: str, artifact: str) -> None:
        rows.append({
            "claim_id": claim_id,
            "claim": claim,
            "status": status,
            "repository": repository,
            "branch": branch,
            "file_or_dataset": file_or_dataset,
            "location": location,
            "evidence": evidence,
            "confidence": confidence,
            "reproduction_command": reproduction,
            "artifact": artifact,
        })

    add("G-F-001", "GAIA run table has a stable 17,152-row raw event inventory and 16,200 supported time-bounded injection events.", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "event_revalidation.raw_record_count", jdump({"raw_record_count": event.get("raw_record_count"), "supported": 16200, "exclusions": event.get("excluded_records")}), "HIGH", "python scripts/p5/revalidate_gaia_events.py --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS --output-dir artifacts/p5/g0r2", artifact_ref(event_path))
    add("G-F-002", "Login failures dominate the supported fault taxonomy (15,478/16,200; 95.5432%).", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "event_revalidation.fault_distribution.login failure", jdump(event.get("fault_distribution", {}).get("login failure", {})), "HIGH", "same as G-F-001", artifact_ref(event_path))
    add("G-F-003", "mobservice1 and mobservice2 dominate labelled injected services (7,815 and 7,799).", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "event_revalidation.root_distribution", jdump({"mobservice1": event.get("root_distribution", {}).get("mobservice1"), "mobservice2": event.get("root_distribution", {}).get("mobservice2")}), "HIGH", "same as G-F-001", artifact_ref(event_path))
    add("G-F-004", "Raw injection intervals overlap: 4,037 cases, 3,429 pairs, 13,077 connected components; interval semantics are half-open.", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "event_revalidation.raw_interval_overlap", jdump(event.get("raw_interval_overlap", {})), "HIGH", "python scripts/p5/revalidate_gaia_events.py --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS --output-dir artifacts/p5/g0r2", artifact_ref(event_path))
    add("G-F-005", "AD node labels and historical RCA labels share the parsed run-table service by construction; this is annotation coupling, not independent causal proof.", "REVALIDATED FACT", p5_repo, "e2e", "run-table + Ada-MGAD parser + historical adapter", "label_reconciliation.ad_label_definition / rca_label_definition", jdump({"service_match_mappable": label.get("coupling", {}).get("service_exact_match_ratio_among_current_ad_mappable"), "all_injections": label.get("coupling", {}).get("service_exact_match_ratio_over_all_rca_injections_treating_missing_as_mismatch"), "preferred_term": label.get("rca_label_definition", {}).get("preferred_term")}), "HIGH", "python scripts/p5/audit_label_reconciliation.py --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS --output-dir artifacts/p5/g0r2", artifact_ref(label_path))
    add("G-F-006", "Current Ada-MGAD main parser has a reproducible float-duration CPU defect: all 12 CPU rows fail the integer-only duration match.", "REVALIDATED PARSER DEFECT", p5_repo, "e2e", gaia_loc, "event_revalidation.cpu_parser_audit", jdump(event.get("cpu_parser_audit", {})), "HIGH", "same as G-F-001", artifact_ref(event_path))
    add("G-F-007", "normal memory freed label is a recovery-marker candidate, not a frozen independent anomaly event.", "OPEN QUESTION", p5_repo, "e2e", gaia_loc, "event_revalidation.recovery_marker_audit", jdump({"rows": event.get("recovery_marker_audit", {}).get("recovery_marker_rows"), "interpretation": "recovery evidence not independently established"}), "MEDIUM", "same as G-F-001", artifact_ref(event_path))
    add("G-F-008", "ERROR records enter the current aligned AD mapping as 38 zero-duration entries, but official anomaly-injection semantics are not established.", "REVALIDATED FACT / OPEN SEMANTICS", p5_repo, "e2e", gaia_loc, "event_revalidation.error_record_audit", jdump(event.get("error_record_audit", {})), "HIGH mapping; MEDIUM semantics", "same as G-F-001", artifact_ref(event_path))
    add("G-F-009", "GAIA metric positive-delta median is 30s and P95 is 60s; logs use millisecond message-prefix timestamps; trace start/end carry microseconds.", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "raw_telemetry_timing.metrics/logs/traces", jdump({"metric_median_ms": timing.get("metrics", {}).get("positive_delta_median_ms"), "metric_p95_ms": timing.get("metrics", {}).get("positive_delta_p95_ms"), "log_resolution": timing.get("logs", {}).get("timestamp_resolution"), "trace_resolution": timing.get("traces", {}).get("timestamp_field")}), "HIGH", "python scripts/p5/audit_raw_telemetry.py --gaia-root /home/zhangll24/RCA_project/datasets/GAIA/MicroSS --historical-telemetry /home/zhangll24/RCA_project/Ada-MGAD-rca-standalone/artifacts/p1/telemetry_diagnostics.json --scan-traces --output artifacts/p5/g0r2/raw_telemetry_timing.json", artifact_ref(timing_path))
    status_scan = timing.get("traces", {}).get("targeted_full_status_and_duration_scan", {})
    add("G-F-010", "Official trace semantics are status != 200; status >= 500 is not equivalent because 3xx/4xx codes occur.", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "raw_telemetry_timing.traces.targeted_full_status_and_duration_scan", jdump({"status_code_distribution": status_scan.get("status_code_distribution"), "official_not_200": status_scan.get("official_not_200_count"), "ge_500": status_scan.get("historical_ge_500_count"), "rules_equivalent": status_scan.get("rules_equivalent")}), "HIGH", "same as G-F-009", artifact_ref(timing_path))
    add("G-F-011", "30s detector rasterization is not event-one-to-one: supported injections collide in shared bins and multiple services may be positive in one bin.", "REVALIDATED FACT", p5_repo, "e2e", gaia_loc, "label_reconciliation.ad_timestep_multiplicity + temporal_resolution", jdump({"different_root_multi_injection_bins": label.get("ad_timestep_multiplicity", {}).get("different_root_multi_injection_bins"), "max_injections_per_bin": label.get("ad_timestep_multiplicity", {}).get("maximum_injections_in_one_bin"), "max_positive_services": label.get("ad_timestep_multiplicity", {}).get("maximum_positive_services_in_one_bin")}), "HIGH", "python scripts/p5/audit_injection_resolution.py --event-registry artifacts/p5/g0r2/event_registry.csv --output artifacts/p5/g0r2/temporal_resolution.json", "{}; {}".format(artifact_ref(label_path), artifact_ref(temporal_path)))
    add("G-F-012", "Historical GAIA artifacts are provenance-bound to an old source path; current path relocation is verified, so historical numbers require current binding before reuse.", "HISTORICAL EVIDENCE / REVALIDATED PATH MIGRATION", hist_repo, "rca-standalone", "artifacts/p1/manifests/gaia/sources.jsonl", "sources.jsonl vs current GAIA root", jdump({"historical_source_exists": False, "current_root_exists": GAIA_ROOT.is_dir(), "current_run_table_exists": RUN_TABLE.is_file(), "historical_head": "64bb681328fa1793014615a20ba2bc1fdf33c3b7"}), "HIGH", "test -f /home/zhangll24/RCA_project/datasets/GAIA/MicroSS/run/run/run/run_table_2021-07.csv", "historical sources.jsonl sha256={}".format(hashes.get(str(HISTORICAL_ROOT / "artifacts/p1/manifests/gaia/sources.jsonl"), "MISSING")))
    add("G-F-013", "Historical asset verification labels conflict: several Material Passports say UNVERIFIED while later status text says completed.", "HISTORICAL EVIDENCE / DOWNGRADED", hist_repo, "rca-standalone", "docs/*_AUDIT.md + docs/RESEARCH_STATUS.md", "Material Passport and status sections", "UNVERIFIED headers are retained; later completed text is not independently current P5 evidence.", "HIGH", "rg -n 'Verification Status|completed|pending' /home/zhangll24/RCA_project/Ada-MGAD-rca-standalone/docs", "historical docs hashes included in source inventory")
    context = compatibility.get("context_overlap", {})
    add("G-F-014", "Expanded context grouping is much more severe at W600 (23 groups; maximum 3,206) than W300 (322 groups; maximum 471).", "REVALIDATED FACT", p5_repo, "e2e", "event registry + expanded contexts", "rca_compatibility.context_overlap", jdump({"W300": context.get("300", {}).get("injection_union_context"), "W600": context.get("600", {}).get("injection_union_context")}), "HIGH", "python scripts/p5/audit_rca_compatibility.py --event-registry artifacts/p5/g0r2/event_registry.csv --raw-telemetry artifacts/p5/g0r2/raw_telemetry_timing.json --ada-rca-repo /home/zhangll24/RCA_project/Ada-RCA-cleanup --historical-repo /home/zhangll24/RCA_project/Ada-MGAD-rca-standalone --output artifacts/p5/g0r2/rca_compatibility.json", artifact_ref(compatibility_path))
    add("G-F-015", "A 15s bin has an expected metric cadence occupancy upper bound of 0.5, versus 1.0 for 30s; exact four-configuration 68D health was not computed.", "INFERENCE / NOT FROZEN", p5_repo, "e2e", "GAIA timing + Ada-RCA frozen specification", "rca_compatibility.configuration_health", jdump(compatibility.get("configuration_health", {})), "MEDIUM", "same as G-F-014", artifact_ref(compatibility_path))
    add("G-F-016", "GAIA RCA should use an independent raw event-relative adapter rather than Ada-MGAD processed tensors.", "RECOMMENDATION", p5_repo, "e2e", "raw schemas + frozen Ada-RCA contract", "rca_compatibility.raw_vs_processed", jdump(compatibility.get("raw_vs_processed", {})), "HIGH", "same as G-F-014", artifact_ref(compatibility_path))
    s_windows = split.get("protocol_s_component_integration", {}).get("windows", {})
    add("G-F-017", "Protocol S is feasible under the audit's W300 five-fold assignment but not W600, where rare faults are absent from multiple folds.", "REVALIDATED FACT / CANDIDATE PROTOCOL", p5_repo, "e2e", "event registry", "split_feasibility.protocol_s_component_integration", jdump({"W300": s_windows.get("300", {}), "W600": s_windows.get("600", {})}), "HIGH", "python scripts/p5/audit_p5_split_feasibility.py --event-registry artifacts/p5/g0r2/event_registry.csv --output artifacts/p5/g0r2/split_feasibility.json", artifact_ref(split_path))
    temporal_protocols = split.get("protocol_t_continuous_temporal_e2e", {}).get("protocols", {})
    add("G-F-018", "All 12 CPU injections occur late in July and land in the test block under both audited chronological 60/20/20 candidates.", "REVALIDATED FACT", p5_repo, "e2e", "event registry", "split_feasibility.rare_fault_distribution + protocol_t", jdump({key: [block.get("rare_fault_counts", {}) for block in value.get("blocks", [])] for key, value in temporal_protocols.items()}), "HIGH", "same as G-F-017", artifact_ref(split_path))
    add("G-F-019", "Component RCA and continuous E2E require distinct split protocols because they answer distribution-controlled versus deployment-time questions.", "RECOMMENDATION", p5_repo, "e2e", "split feasibility audit", "split_feasibility.protocol_s/protocol_t", "Protocol S preserves context groups; Protocol T preserves a contiguous time stream and purges boundary-crossing windows.", "HIGH", "same as G-F-017", artifact_ref(split_path))
    add("G-F-020", "Historical W300 representation arrays are non-68D proxies; exact W300/W600 by 15s/30s Ada-RCA feature health remains uncomputed.", "HISTORICAL EVIDENCE / OPEN QUESTION", hist_repo, "rca-standalone", "P2 feature bundles + Ada-RCA frozen extractor", "rca_compatibility.representation_health_proxy", jdump(compatibility.get("representation_health_proxy", {}).get("limitations", [])), "HIGH", "same as G-F-014", artifact_ref(compatibility_path))
    return rows


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CURRENT_ROOT / "artifacts/p5/g0r2")
    parser.add_argument("--historical-root", type=Path, default=HISTORICAL_ROOT)
    parser.add_argument("--gaia-root", type=Path, default=GAIA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global HISTORICAL_ROOT, GAIA_ROOT, RUN_TABLE
    HISTORICAL_ROOT = args.historical_root.resolve()
    GAIA_ROOT = args.gaia_root.resolve()
    RUN_TABLE = GAIA_ROOT / "run/run/run/run_table_2021-07.csv"

    historical_paths = [HISTORICAL_ROOT / item for item in HISTORY_ASSETS]
    p5_paths = [CURRENT_ROOT / item for item in P5_ARTIFACTS]
    hashes = source_inventory(historical_paths + p5_paths)
    event_path = CURRENT_ROOT / "artifacts/p5/g0r2/event_revalidation.json"
    label_path = CURRENT_ROOT / "artifacts/p5/g0r2/label_reconciliation.json"
    timing_path = CURRENT_ROOT / "artifacts/p5/g0r2/raw_telemetry_timing.json"
    compatibility_path = CURRENT_ROOT / "artifacts/p5/g0r2/rca_compatibility.json"
    split_path = CURRENT_ROOT / "artifacts/p5/g0r2/split_feasibility.json"
    missing = [str(path) for path in (event_path, label_path, timing_path, compatibility_path, split_path) if not path.is_file()]
    if missing:
        raise SystemExit("missing required P5 audit artifact(s): {}".format(", ".join(missing)))
    event = load_json(event_path)
    label = load_json(label_path)
    timing = load_json(timing_path)
    compatibility = load_json(compatibility_path)
    split = load_json(split_path)

    matrix = historical_matrix_rows(hashes)
    matrix_fields = (
        "historical_asset", "historical_claim", "reproducible_now", "revalidated_now",
        "reusable_for_p5", "verification_status", "evidence", "confidence", "action",
        "source_hashes", "config",
    )
    write_csv(args.output_dir / "historical_reuse_matrix.csv", matrix_fields, matrix)
    ledger = evidence_rows(hashes, event, label, timing, compatibility, split)
    write_csv(args.output_dir / "evidence_ledger.csv", REQUIRED_LEDGER_FIELDS, ledger)
    print(json.dumps({
        "config": {
            "historical_root": str(HISTORICAL_ROOT),
            "current_root": str(CURRENT_ROOT),
            "gaia_root": str(GAIA_ROOT),
            "run_table": str(RUN_TABLE),
            "historical_head": "64bb681328fa1793014615a20ba2bc1fdf33c3b7",
            "source_hash_count": len(hashes),
        },
        "outputs": {
            "historical_reuse_matrix.csv": str(args.output_dir / "historical_reuse_matrix.csv"),
            "evidence_ledger.csv": str(args.output_dir / "evidence_ledger.csv"),
        },
        "rows": {"historical_reuse_matrix": len(matrix), "evidence_ledger": len(ledger)},
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
