"""V2 start evidence; the V1 forecaster, training and screen remain fixed."""

import ast
import json
from pathlib import Path

from scripts.p6.analyze_c0_causal_development import verify_completion
from src.e2e.protocol import sha256_file


PROTOCOL_ID = "P6-NORMAL-FORECAST-FIT-SCREEN-STABILITY-V2"
GATE_NAMES = {"precision_floor", "recall_gain", "f1_gain", "single_onset_recall_decline_limit"}


def validate_fallback_config(config):
    spec = config["stability_fallback"]
    reference = json.loads(Path(spec["reference_config"]).read_text())
    if reference["protocol_id"] != "P6-NORMAL-FORECAST-FIT-SCREEN-V1":
        raise ValueError("fallback must preserve the previously registered Fit screen")
    allowed = {"protocol_id", "start_condition", "stability_fallback"}
    if ({key: value for key, value in config.items() if key not in allowed}
            != {key: value for key, value in reference.items() if key not in allowed}
            or config["protocol_id"] != PROTOCOL_ID):
        raise ValueError("fallback changes the forecaster beyond its startup evidence")


def verify_forecast_kernel(current, reference):
    """The complete main body and all numeric/scoring functions are unchanged."""
    def functions(path):
        tree = ast.parse(Path(path).read_text())
        return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    now, old = functions(current), functions(reference)
    for name in ("main", "cohort_summary", "loader", "export_scores", "real_prediction_gate", "screen_results"):
        if now[name] != old[name]:
            raise ValueError("frozen forecast execution changed: " + name)


def stability_start_bindings(config, root):
    validate_fallback_config(config)
    spec = config["stability_fallback"]
    directory = Path(spec["run"])
    verify_completion(directory)
    manifest_path = directory / "completion_manifest.json"
    report_path = directory / "replication_comparison.json"
    if (sha256_file(manifest_path) != spec["completion_sha256"]
            or sha256_file(report_path) != spec["report_sha256"]):
        raise ValueError("stability decision differs from the registered fallback evidence")
    report = json.loads(report_path.read_text())
    lock = json.loads((directory / "analysis_input_lock.json").read_text())
    expected_spec = {"seeds": [17, 2026], "primary_seed": 42,
        "threshold_rule": "each seed's unchanged merged selector, frozen before bin decoding",
        "time_strata": "two equal duration halves of the frozen Validation block",
        "minimum_half_recall_gain": .03, "all_seeds_must_pass": True, "best_seed_selection": False}
    if (report["status"] != "STABILITY_NO_GO" or report["spec"] != expected_spec
            or set(report["seeds"]) != {"42", "17", "2026"}
            or report["primary_seed"] != 42 or report["best_seed_selection"] is not False
            or report["gt_denominator"] != 2901 or report["test_arrays_read"] is not False
            or report["test_inference_run"] is not False or report["full_e2e_run"] is not False
            or report["seeds"]["42"]["passed"] is not True
            or any(report["seeds"][str(seed)]["passed"] is not False for seed in (17, 2026))):
        raise ValueError("not a complete initialization-stability failure")
    bindings = dict(lock["source_sha256"])
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("stability source/member binding drift: " + path)
    protocol_paths = [Path(path) for path in bindings
                      if path.endswith("/configs/e2e/gaia_p6_c0_tcn_replication_v1.json")]
    if (len(protocol_paths) != 1
            or json.loads(protocol_paths[0].read_text())["protocol_id"] != "P6-C0-ONSET30-TCN-REPLICATION-V1"):
        raise ValueError("stability result has no bound replication protocol")
    for seed in (17, 2026):
        decoder = directory.parent / ("seed%d-bin-v1" % seed)
        model = directory.parent / ("seed%d-v1" % seed)
        for member in (decoder, model):
            verify_completion(member)
            path = member / "completion_manifest.json"
            if bindings.get(str(path.resolve())) != sha256_file(path):
                raise ValueError("stability does not bind the declared member")
        selection = json.loads((model / "validation_selection.json").read_text())
        decoded = json.loads((decoder / "decoder_comparison.json").read_text())
        if (selection["model_args"]["random_seed"] != seed
                or decoded["model_completion_sha256"] != sha256_file(model / "completion_manifest.json")
                or decoded["gt_denominator"] != 2901 or decoded["full_e2e_run"]
                or any(set(gates) != GATE_NAMES or all(gates.values())
                       for gates in decoded["gates"].values())
                or set(decoded["gates"]) != {"onset_bins_fixed_threshold", "onset_bins_validation_selected"}):
            raise ValueError("a replication decoder has not failed both registered variants")
    refs = {name: Path(path) for name, path in config["references"].items()}
    for member in refs.values():
        verify_completion(member)
        path = member / "completion_manifest.json"
        bindings[str(path.resolve())] = sha256_file(path)
    primary = json.loads((refs["tcn_decoder_comparison"] / "decoder_comparison.json").read_text())
    if (primary["model_completion_sha256"] != sha256_file(refs["tcn_run"] / "completion_manifest.json")
            or primary["results"]["onset_bins_fixed_threshold"]["metrics"] != report["seeds"]["42"]["metrics"]):
        raise ValueError("stability primary is not the originally registered candidate")
    verify_forecast_kernel(root / "scripts/p6/run_normal_forecast_fit_screen.py", spec["reference_runner"])
    if sha256_file(root / "src/e2e/normal_forecast.py") != sha256_file(Path(spec["reference_runner"]).parents[2] / "src/e2e/normal_forecast.py"):
        raise ValueError("fallback changes the frozen forecaster or Fit scope")
    for path in (manifest_path, report_path, directory / "analysis_input_lock.json",
                 Path(spec["reference_config"]), Path(spec["reference_runner"]),
                 root / "scripts/p6/normal_forecast_stability_fallback.py",
                 root / "docs/P6_NORMAL_FORECAST_STABILITY_FALLBACK_V2.md"):
        bindings[str(path.resolve())] = sha256_file(path)
    return bindings
