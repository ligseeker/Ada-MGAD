import json
from pathlib import Path
import tempfile
import unittest

from src.e2e.experiment_layout import ExperimentLayout


class ExperimentLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.project = Path(self.tempdir.name)
        self.config = {
            "shared_inputs": {
                "ad_data_root": "shared/ad-data",
                "ad_preprocess_artifact_root": "shared/ad-preprocess",
                "protocol_root": "shared/protocol",
                "rca_index_root": "shared/rca-index",
                "rca_gt_feature_root": "shared/rca-gt-features",
                "rca_gt_artifact_root": "shared/rca-gt-artifacts",
                "rca_gt_case_registry": "shared/rca-gt-artifacts/rca_case_registry_gt.csv",
            }
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def test_resolve_is_side_effect_free_and_outputs_are_run_local(self):
        layout = ExperimentLayout.resolve(self.project, self.config, "runs/example")
        self.assertEqual(layout.run_dir, self.project / "runs/example")
        self.assertFalse(layout.run_dir.exists())
        for name, value in layout.as_dict().items():
            if name.endswith("_root") or name.endswith("_path") or name in {
                "run_dir", "lock_path", "state_path", "rca_detected_case_registry",
            }:
                path = Path(value)
                if name in {
                    "project_root",
                    "ad_data_root", "ad_preprocess_artifact_root", "protocol_root",
                    "rca_index_root", "rca_gt_feature_root", "rca_gt_artifact_root",
                    "rca_gt_case_registry",
                }:
                    continue
                self.assertTrue(path == layout.run_dir or layout.run_dir in path.parents)

    def test_create_load_and_state_machine(self):
        layout = ExperimentLayout.resolve(self.project, self.config, "runs/example")
        state = layout.create_new({"config_sha256": "abc", "seed": 42})
        self.assertEqual(state["status"], "INITIALIZED")
        self.assertEqual(json.loads(layout.state_path.read_text())["status"], "INITIALIZED")
        for output in (
            layout.logs_root, layout.ad_artifact_root, layout.ad_checkpoint_root,
            layout.event_root, layout.rca_artifact_root,
            layout.rca_detected_artifact_root, layout.rca_detected_feature_root,
        ):
            self.assertTrue(output.is_dir())
        self.assertEqual(layout.load_existing()["status"], "INITIALIZED")

        layout.transition("INITIALIZED", "AD_RUNNING", {"pid": 123})
        layout.transition("AD_RUNNING", "AD_COMPLETE")
        layout.transition("AD_COMPLETE", "POST_AD_RUNNING")
        state = layout.transition("POST_AD_RUNNING", "COMPLETE", {"cases": 2})
        self.assertEqual(state["status"], "COMPLETE")
        self.assertEqual(state["details"], {"cases": 2})
        self.assertEqual(len(state["transitions"]), 4)
        with self.assertRaises(ValueError):
            layout.transition("COMPLETE", "AD_RUNNING")
        with self.assertRaises(FileExistsError):
            layout.create_new({})

    def test_failed_edges_and_mismatched_state_are_rejected(self):
        layout = ExperimentLayout.resolve(self.project, self.config, "runs/failing")
        layout.create_new({})
        with self.assertRaises(ValueError):
            layout.transition("AD_RUNNING", "AD_COMPLETE")
        layout.transition("INITIALIZED", "AD_RUNNING")
        layout.transition("AD_RUNNING", "FAILED")
        with self.assertRaises(ValueError):
            layout.transition("FAILED", "COMPLETE")

        raw = json.loads(layout.state_path.read_text())
        raw["run_dir"] = str(self.project / "runs/other")
        layout.state_path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            layout.load_existing()

    def test_illegal_run_directory_and_shared_overlap_are_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentLayout.resolve(self.project, self.config, self.project.parent / "outside")
        with self.assertRaises(ValueError):
            ExperimentLayout.resolve(
                self.project,
                {"shared_inputs": {"ad_data_root": "runs/example"}},
                "runs/example",
            )
        with self.assertRaises(ValueError):
            ExperimentLayout.resolve(
                self.project,
                {"shared_inputs": {"ad_data_root": "shared"}},
                "shared/runs/example",
            )

    def test_gt_case_registry_default_is_shared_v3_path(self):
        layout = ExperimentLayout.resolve(self.project, {}, "runs/default")
        self.assertEqual(
            layout.rca_gt_case_registry,
            (self.project / "artifacts/p5/v3/rca/rca_case_registry_gt.csv").resolve(),
        )
        self.assertEqual(
            layout.rca_gt_feature_root,
            (self.project / "data/p5/v3/rca_features_gt").resolve(),
        )
        self.assertEqual(
            layout.rca_gt_artifact_root,
            (self.project / "artifacts/p5/v3/rca_gt_features").resolve(),
        )
        self.assertEqual(
            layout.as_dict()["rca_gt_case_registry"],
            str(layout.rca_gt_case_registry),
        )


if __name__ == "__main__":
    unittest.main()
