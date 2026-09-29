"""P6-C1-v2 runner: exclusive sealed stages and run-lock drift detection."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p6.run_c1_v2 import _stage, _validate_stage


class C1V2RunnerStageTest(unittest.TestCase):
    def test_stage_is_exclusive_sealed_and_never_reopened(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_lock.json").write_text(
                json.dumps({"source_sha256": {"mock": "fixed"}}), encoding="utf-8")

            def work(directory):
                (directory / "output.txt").write_text("sealed\n", encoding="utf-8")
                return "COMPLETE", ("output.txt",), {"input": "hash"}, {"n": 1}

            with patch("scripts.p6.run_c1_v2.source_hashes", return_value={"mock": "fixed"}):
                result = _stage(root, "predictions", work)
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(_validate_stage(root, "predictions")["details"], {"n": 1})
                with self.assertRaises(FileExistsError):
                    _stage(root, "predictions", work)
                (root / "predictions/output.txt").write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "output drift"):
                    _validate_stage(root, "predictions")

    def test_no_go_stage_is_refused_by_require_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_lock.json").write_text(
                json.dumps({"source_sha256": {"mock": "fixed"}}), encoding="utf-8")

            def work(directory):
                (directory / "cases.csv").write_text("case_id\n", encoding="utf-8")
                return "NO_GO", ("cases.csv",), {}, {"floors_pass": False}

            with patch("scripts.p6.run_c1_v2.source_hashes", return_value={"mock": "fixed"}):
                manifest = _stage(root, "train_cohort", work)
                self.assertEqual(manifest["status"], "NO_GO")
                self.assertEqual(
                    _validate_stage(root, "train_cohort", require_complete=False)["status"],
                    "NO_GO")
                with self.assertRaisesRegex(ValueError, "NO_GO"):
                    _validate_stage(root, "train_cohort")


if __name__ == "__main__":
    unittest.main()
