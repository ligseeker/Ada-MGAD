import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.e2e.experiment_layout import ExperimentLayout
from scripts.p5 import run_i1_pipeline as pipeline


class ExperimentPipelineTests(unittest.TestCase):
    """Small contract tests for the orchestration shell, not the data path."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.project = Path(self.tempdir.name) / "project"
        self.project.mkdir()
        shared = self.project / "shared"
        shared.mkdir()

        self.config = {
            "random_seed": 42,
            "event_registry": {"path": str(shared / "event_registry.csv")},
            "ad_preprocessing": {
                "frozen_schema_path": str(shared / "schema.json"),
            },
            "shared_inputs": {
                "ad_data_root": str(shared / "ad-data"),
                "ad_preprocess_artifact_root": str(shared / "ad-preprocess"),
                "protocol_root": str(shared / "protocol"),
                "rca_index_root": str(shared / "rca-index"),
                "rca_gt_feature_root": str(shared / "rca-gt-features"),
                "rca_gt_artifact_root": str(shared / "rca-gt-artifacts"),
                "rca_gt_case_registry": str(
                    shared / "rca-gt-artifacts" / "rca_case_registry_gt.csv"
                ),
            },
        }
        self.config_path = self.project / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

        # These are immutable inputs to a run.  Their contents are irrelevant
        # here; existence is enough for the orchestration gate.
        for path in (
            shared / "event_registry.csv",
            shared / "schema.json",
            shared / "ad-preprocess" / "ad_data_manifest.json",
            shared / "protocol" / "protocol_manifest.json",
            shared / "protocol" / "split_manifest.json",
            shared / "protocol" / "provenance.json",
            shared / "rca-index" / "index_manifest.json",
            shared / "rca-gt-artifacts" / "rca_feature_manifest.json",
            shared / "rca-gt-artifacts" / "rca_case_registry_gt.csv",
            shared / "rca-gt-features" / "z2_features.npy",
            shared / "rca-gt-features" / "case_ids.npy",
            shared / "rca-gt-features" / "splits.npy",
            shared / "rca-gt-features" / "anchors_ms.npy",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        (shared / "ad-data").mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def _layout(self, name):
        return ExperimentLayout.resolve(
            self.project, self.config, self.project / "runs" / name
        )

    def _args(self, action="ad-train"):
        return SimpleNamespace(
            action=action,
            config=str(self.config_path),
            run_dir=None,
            gpu=False,
            raw_root=None,
            chunk_rows=None,
            raw_workers=None,
            ad_workers=None,
            feature_workers=3,
            event_workers=4,
            case_chunk_size=7,
            start_method="spawn",
        )

    def _prepare(self, layout, action="ad-train", require_post_inputs=False):
        args = self._args(action)
        with patch.object(pipeline, "_config_path", return_value=self.config_path), \
                patch.object(pipeline, "_git_head", return_value="test-head"), \
                patch.object(pipeline, "_tracked_worktree_dirty", return_value=False):
            pipeline._prepare_new_run(
                args, self.config, layout, require_post_inputs=require_post_inputs
            )
        return args

    def _make_ad_outputs(self, layout):
        for path in (
            layout.ad_checkpoint_root / "best_train_f1.pt",
            layout.ad_checkpoint_root / "best_train_loss.pt",
            layout.ad_checkpoint_root / "last.pt",
            layout.ad_artifact_root / "ad_training_summary.json",
            layout.ad_artifact_root / "ad_train_predictions.csv",
            layout.ad_artifact_root / "ad_test_predictions.csv",
            layout.ad_artifact_root / "reconstruction_calibration.json",
        ):
            path.write_text("fixture\n", encoding="utf-8")

    def test_runs_have_distinct_mutable_directories_and_initial_state(self):
        first = self._layout("first")
        second = self._layout("second")
        self._prepare(first)
        self._prepare(second)

        self.assertNotEqual(first.run_dir, second.run_dir)
        self.assertNotEqual(first.ad_artifact_root, second.ad_artifact_root)
        self.assertNotEqual(first.lock_path, second.lock_path)
        self.assertEqual(first.read_state()["status"], "INITIALIZED")
        self.assertEqual(second.read_state()["status"], "INITIALIZED")
        self.assertEqual(json.loads(first.state_path.read_text())["metadata"]["run_id"], "first")
        self.assertEqual(json.loads(second.state_path.read_text())["metadata"]["run_id"], "second")

    def test_run_helper_tees_child_output_to_stage_and_pipeline_logs(self):
        stage_log = self.project / "runs" / "one" / "logs" / "01_ad_train.log"
        pipeline_log = self.project / "runs" / "one" / "logs" / "pipeline.log"

        class FakeProcess:
            stdout = iter(("child stdout\n", "child stderr merged\n"))

            def poll(self):
                return 0

            def wait(self):
                return 0

        captured = io.StringIO()
        with patch.object(pipeline.subprocess, "Popen", return_value=FakeProcess()), \
                contextlib.redirect_stdout(captured):
            pipeline._run(
                ["scripts/p5/fake_stage.py", "--run", "runs/one"],
                (stage_log, pipeline_log),
            )

        self.assertIn("child stdout", captured.getvalue())
        self.assertIn("child stderr merged", captured.getvalue())
        self.assertEqual(stage_log.read_text(encoding="utf-8"), pipeline_log.read_text(encoding="utf-8"))
        self.assertIn("RUN", stage_log.read_text(encoding="utf-8"))
        self.assertIn("child stdout", stage_log.read_text(encoding="utf-8"))

    def test_ad_stage_injects_only_run_local_paths_and_migrates_state(self):
        layout = self._layout("ad")
        args = self._prepare(layout)
        self._make_ad_outputs(layout)
        calls = []

        with patch.object(pipeline, "_run", side_effect=lambda command, logs=(): calls.append((command, logs))):
            pipeline._run_ad_stage(args, layout)

        self.assertEqual(layout.read_state()["status"], "AD_COMPLETE")
        self.assertEqual(len(calls), 1)
        command = calls[0][0]
        self.assertIn("--data-root", command)
        self.assertEqual(command[command.index("--data-root") + 1], str(layout.ad_data_root))
        self.assertEqual(
            command[command.index("--preprocess-artifact-root") + 1],
            str(layout.ad_preprocess_artifact_root),
        )
        self.assertEqual(command[command.index("--artifact-root") + 1], str(layout.ad_artifact_root))
        self.assertEqual(command[command.index("--checkpoint-dir") + 1], str(layout.ad_checkpoint_root))
        self.assertIn(str(layout.logs_root / "01_ad_train.log"), [str(path) for path in calls[0][1]])
        self.assertNotIn("data/p5/v3", " ".join(command))
        self.assertNotIn("artifacts/p5/v3", " ".join(command))

    def test_ad_stage_failure_records_failed_state(self):
        layout = self._layout("ad-failed")
        args = self._prepare(layout)
        with patch.object(pipeline, "_run", side_effect=RuntimeError("mock child failure")):
            with self.assertRaisesRegex(RuntimeError, "mock child failure"):
                pipeline._run_ad_stage(args, layout)
        state = layout.read_state()
        self.assertEqual(state["status"], "FAILED")
        self.assertEqual(state["details"]["stage"], "ad-train")

    def test_post_ad_stage_injects_run_paths_and_completes_state(self):
        layout = self._layout("post")
        args = self._prepare(layout)
        self._make_ad_outputs(layout)
        layout.transition("INITIALIZED", "AD_RUNNING")
        layout.transition("AD_RUNNING", "AD_COMPLETE")
        # The post-AD gate and finalizer gate are both represented without
        # launching any real child process.
        for path in (
            layout.run_dir / "run_manifest.json",
            layout.run_dir / "final_report.md",
        ):
            path.write_text("fixture\n", encoding="utf-8")
        calls = []
        with patch.object(pipeline, "_git_head", return_value="test-head"), \
                patch.object(pipeline, "_tracked_worktree_dirty", return_value=False), \
                patch.object(
                    pipeline, "_run",
                    side_effect=lambda command, logs=(): calls.append((command, logs)),
                ):
            pipeline._run_post_ad_stage(args, layout)

        self.assertEqual(layout.read_state()["status"], "COMPLETE")
        self.assertEqual(len(calls), 6)
        rendered = " ".join(" ".join(str(value) for value in command) for command, _ in calls)
        for path in (
            layout.event_root,
            layout.ad_artifact_root,
            layout.rca_detected_case_registry,
            layout.rca_detected_feature_root,
            layout.rca_detected_artifact_root,
            layout.rca_model_path,
            layout.rca_artifact_root,
            layout.rca_index_root,
        ):
            self.assertIn(str(path), rendered)
        self.assertIn("--run-dir", calls[-1][0])
        self.assertEqual(calls[-1][0][calls[-1][0].index("--run-dir") + 1], str(layout.run_dir))
        self.assertTrue(all(str(layout.logs_root) in str(log) for _, logs in calls for log in logs))
        self.assertNotIn("data/p5/v3", rendered)
        self.assertNotIn("artifacts/p5/v3", rendered)


if __name__ == "__main__":
    unittest.main()
