"""P6-C1-v2 evaluation stage: the sealed-lock to C1/C2 chain runs end to end.

The formal run binds its source hashes, so this synthetic dry-run of the
evaluation stage is the guard that the post-lock chain works before the real
Test lock exists. It uses only synthetic frames; no frozen Test GT is read.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.p6 import run_c1_v2 as runner
from src.e2e import c1_evaluation as source_evaluation
from src.e2e import c1_v2_c2 as c2_module
from src.e2e import c1_v2_shared_data as shared
from src.e2e import rca_model as source_rca_model
from src.e2e.c1_v2_c2 import LEDGER_CATEGORIES

START = 1626963120000


def _ranking_at(service: str, position: int) -> str:
    others = [item for item in shared.GAIA_SERVICES if item != service]
    order = others[:position - 1] + [service] + others[position - 1:]
    return json.dumps(order, separators=(",", ":"))


class _StubArm:
    def scores(self, features):
        return np.zeros(len(shared.GAIA_SERVICES))


def _write_stage(root: Path, name: str, files, inputs, details):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "p6_c1_v2_stage_manifest_v1", "stage": name, "status": "COMPLETE",
        "run_lock_sha256": shared.sha256(root / "run_lock.json"),
        "source_sha256": shared.source_hashes(), "inputs": inputs, "details": details,
        "files": {filename: {"bytes": (directory / filename).stat().st_size,
                             "sha256": shared.sha256(directory / filename)}
                  for filename in files},
    }
    with (directory / "completion_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


class C1V2EvaluationStageTest(unittest.TestCase):
    def _fixture(self, root: Path):
        episodes = pd.DataFrame([
            {"prediction_id": "test-pred-000000", "split": "test", "t_hat": START + 600000,
             "episode_end_time": START + 630000, "positive_bins": 1, "system_score": 0.99,
             "threshold": 0.9},
            {"prediction_id": "test-pred-000001", "split": "test", "t_hat": START + 1200000,
             "episode_end_time": START + 1230000, "positive_bins": 1, "system_score": 0.98,
             "threshold": 0.9},
            {"prediction_id": "test-pred-000002", "split": "test", "t_hat": START + 1800000,
             "episode_end_time": START + 1830000, "positive_bins": 1, "system_score": 0.97,
             "threshold": 0.9},
        ])
        matching = pd.DataFrame([
            {"split": "test", "prediction_id": "test-pred-000000", "t_hat": START + 600000,
             "episode_end_time": START + 630000, "positive_bins": 1, "system_score": 0.99,
             "threshold": 0.9, "match_status": "matched", "case_id": "gaia-v3-90001",
             "source_index": 90001, "gt_service": "mobservice1", "fault_type": "login_failure",
             "gt_start_ms": START + 570000, "gt_end_ms": START + 590000,
             "detection_delay_seconds": 30.0, "absolute_onset_error_seconds": 30.0},
            {"split": "test", "prediction_id": "test-pred-000001", "t_hat": START + 1200000,
             "episode_end_time": START + 1230000, "positive_bins": 1, "system_score": 0.98,
             "threshold": 0.9, "match_status": "matched", "case_id": "gaia-v3-90002",
             "source_index": 90002, "gt_service": "mobservice2", "fault_type": "memory_anomalies",
             "gt_start_ms": START + 1140000, "gt_end_ms": START + 1180000,
             "detection_delay_seconds": 60.0, "absolute_onset_error_seconds": 60.0},
            {"split": "test", "prediction_id": "test-pred-000002", "t_hat": START + 1800000,
             "episode_end_time": START + 1830000, "positive_bins": 1, "system_score": 0.97,
             "threshold": 0.9, "match_status": "false_alarm", "case_id": None,
             "source_index": None, "gt_service": None, "fault_type": None, "gt_start_ms": None,
             "gt_end_ms": None, "detection_delay_seconds": None,
             "absolute_onset_error_seconds": None},
            {"split": "test", "prediction_id": None, "t_hat": None, "episode_end_time": None,
             "positive_bins": None, "system_score": None, "threshold": None,
             "match_status": "miss", "case_id": "gaia-v3-90003", "source_index": 90003,
             "gt_service": "dbservice1", "fault_type": "login_failure",
             "gt_start_ms": START + 2400000, "gt_end_ms": START + 2460000,
             "detection_delay_seconds": None, "absolute_onset_error_seconds": None},
        ])
        scope = pd.DataFrame([
            {"prediction_id": "test-pred-000000", "t_hat": START + 600000,
             "scope_status": "legal", "ranking_status": "complete", "failure_reason": "",
             "ranking_b": _ranking_at("mobservice1", 1), "ranking_c": _ranking_at("mobservice1", 2)},
            {"prediction_id": "test-pred-000001", "t_hat": START + 1200000,
             "scope_status": "legal", "ranking_status": "complete", "failure_reason": "",
             "ranking_b": _ranking_at("mobservice2", 2), "ranking_c": _ranking_at("mobservice2", 1)},
            {"prediction_id": "test-pred-000002", "t_hat": START + 1800000,
             "scope_status": "legal", "ranking_status": "complete", "failure_reason": "",
             "ranking_b": _ranking_at("webservice1", 1), "ranking_c": _ranking_at("webservice1", 1)},
        ])
        registry = pd.DataFrame([
            {"case_id": "gaia-v3-90001", "source_index": 90001, "service": "mobservice1",
             "fault_type": "login_failure", "start_ms": START + 570000,
             "end_ms": START + 590000, "detector_domain": True, "split": "test"},
            {"case_id": "gaia-v3-90002", "source_index": 90002, "service": "mobservice2",
             "fault_type": "memory_anomalies", "start_ms": START + 1140000,
             "end_ms": START + 1180000, "detector_domain": True, "split": "test"},
            {"case_id": "gaia-v3-90003", "source_index": 90003, "service": "dbservice1",
             "fault_type": "login_failure", "start_ms": START + 2400000,
             "end_ms": START + 2460000, "detector_domain": True, "split": "test"},
        ])
        base = {"split": {"boundary_ms": START, "absolute_end_ms": START + 3600000}}
        temp = root / "bound"
        temp.mkdir()
        paths = {}
        for name, frame in (("episodes", episodes), ("matching", matching), ("scope", scope)):
            paths[name] = temp / (name + ".csv")
            frame.to_csv(paths[name], index=False, lineterminator="\n")
        paths["predictions"] = temp / "test_predictions.csv"
        paths["predictions"].write_text("prediction_id,trigger_label\nx,1\n", encoding="utf-8")
        paths["metrics"] = temp / "test_metrics.json"
        paths["metrics"].write_text(json.dumps({
            "test_event_metrics": {"event_precision": 0.9962, "event_recall": 0.7254,
                                   "event_f1": 0.8395},
            "frozen_validation_threshold": 0.9998, "selected_epoch": 8,
            "pre_registered_gate": {"precision": 0.9}, "p6_c0_decision": "BORDERLINE"}),
            encoding="utf-8")
        paths["index"] = temp / "index_manifest.json"
        paths["index"].write_text("{}\n", encoding="utf-8")
        paths["base"] = temp / "base_config.json"
        paths["base"].write_text(json.dumps(base), encoding="utf-8")
        protocol = {"bindings": {
            "c0_test_episodes": {"path": str(paths["episodes"])},
            "c0_test_matching": {"path": str(paths["matching"])},
            "c0_test_predictions": {"path": str(paths["predictions"])},
            "c0_test_metrics": {"path": str(paths["metrics"])},
            "rca_index_manifest": {"path": str(paths["index"])},
            "base_config": {"path": str(paths["base"])},
        }}
        return protocol, paths, registry

    def test_post_lock_evaluation_writes_c1_and_c2_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, paths, registry = self._fixture(root)
            (root / "run_lock.json").write_text(
                json.dumps({"source_sha256": shared.source_hashes()}), encoding="utf-8")
            scope = pd.read_csv(paths["scope"], keep_default_na=False)

            def _save(directory, name, frame):
                frame.to_csv(directory / name, index=False, lineterminator="\n")

            # Rebuild the two sealed upstream stages with the runner's own schema.
            (root / "train_cohort").mkdir()
            (root / "train_cohort/completion_manifest.json").write_text(
                json.dumps({"schema_version": "p6_c1_v2_stage_manifest_v1",
                            "stage": "train_cohort", "status": "COMPLETE"}), encoding="utf-8")
            rca_dir = root / "rca"
            rca_dir.mkdir()
            for name in ("arm_b.npz", "arm_c.npz", "shared_scaler.npz"):
                (rca_dir / name).write_bytes(b"stub")
            cohort_manifest_sha = shared.sha256(root / "train_cohort/completion_manifest.json")
            _write_stage(root, "rca", ("arm_b.npz", "arm_c.npz", "shared_scaler.npz"),
                         {"train_cohort_manifest": cohort_manifest_sha}, {"common_cases": 2})
            predictions = root / "predictions"
            predictions.mkdir()
            _save(predictions, "scope_rankings.csv", scope)
            np.save(predictions / "feature_inputs.npy", np.zeros((len(scope), 10, 68), dtype=np.float32))
            np.save(predictions / "feature_valid.npy", np.ones(len(scope), dtype=bool))
            (predictions / "prediction_lock.json").write_text("{}\n", encoding="utf-8")
            _write_stage(root, "predictions",
                         ("scope_rankings.csv", "feature_inputs.npy", "feature_valid.npy",
                          "prediction_lock.json"),
                         {"rca_manifest": shared.sha256(root / "rca/completion_manifest.json"),
                          "raw_index_manifest": shared.sha256(paths["index"]),
                          "c0_test_episodes": shared.sha256(paths["episodes"]),
                          "c0_test_predictions_hash_only": shared.sha256(paths["predictions"])},
                         {"all_episodes": len(scope)})

            def _bound(binding):
                return Path(binding["path"])

            with patch.object(runner, "bound_path", side_effect=_bound), \
                 patch.object(shared, "bound_path", side_effect=_bound), \
                 patch.object(c2_module, "bound_path", side_effect=_bound), \
                 patch.object(runner, "_base", return_value={"split": {
                     "boundary_ms": START, "absolute_end_ms": START + 3600000}}), \
                 patch.object(runner, "load_registry", return_value=registry), \
                 patch.object(source_rca_model, "load_conditional_logit",
                              return_value=_StubArm()), \
                 patch.object(source_evaluation, "evaluate_c1_oracle_a",
                              return_value=(pd.DataFrame([{"case_id": "gaia-v3-90001"}]),
                                            {"n_same_as_bc": 2, "AC@1": 1.0})), \
                 patch.object(runner, "_index", return_value=(object(), paths["index"])):
                runner._evaluate(root, protocol)

            evaluation = root / "evaluation"
            c1 = json.loads((evaluation / "c1_results.json").read_text())
            c2 = json.loads((evaluation / "c2_full_diagnosis.json").read_text())
            stage1 = json.loads((evaluation / "c2_stage1_reference.json").read_text())
            self.assertEqual(c1["primary"]["n"], 2)
            self.assertEqual(c1["primary"]["paired_transitions"],
                             {"both_correct": 0, "b_only_correct": 1, "c_only_correct": 1,
                              "both_incorrect": 0})
            self.assertAlmostEqual(c1["primary"]["delta_c_minus_b"], 0.0)
            self.assertLessEqual(c1["primary"]["onset_day_cluster_bootstrap_95_percentile"]["lower"],
                                 c1["primary"]["onset_day_cluster_bootstrap_95_percentile"]["upper"])
            self.assertFalse(stage1["recomputed"])
            for arm in ("b", "c"):
                self.assertEqual(set(c2["arms"][arm]["ledger_categories"]), set(LEDGER_CATEGORIES))
            ledger = pd.read_csv(evaluation / "c2_failure_ledger.csv")
            self.assertEqual(set(ledger["arm"]), {"b", "c"})
            self.assertTrue((evaluation / "c2_raw_matching.csv").is_file())
            self.assertTrue((evaluation / "oracle_a_rankings.csv").is_file())
            manifest = json.loads((evaluation / "completion_manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertFalse((evaluation / "INCOMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
