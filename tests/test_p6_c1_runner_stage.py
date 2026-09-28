"""C1 runner stages use exclusive creation and reject tampered outputs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p6.run_c1 import _init, _protocol, _run_lock, _stage, _validate_stage


class C1RunnerStageTest(unittest.TestCase):
    def test_init_reserves_one_root_and_pins_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new_run"
            protocol = _protocol()
            with patch("scripts.p6.run_c1._run_root", return_value=root), \
                 patch("scripts.p6.run_c1._preflight"):
                _init(protocol, gpu=False)
                self.assertEqual(_run_lock(root, protocol)["device"], "cpu")
                self.assertTrue((root / "folds").is_dir())
                self.assertFalse((root / "INCOMPLETE.json").exists())
                with self.assertRaisesRegex(ValueError, "already exists"):
                    _init(protocol, gpu=False)

    def test_stage_is_exclusive_and_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run_lock.json").write_text(
                json.dumps({"source_sha256": {"mock": "fixed"}}), encoding="utf-8")

            def work(directory):
                (directory / "output.txt").write_text("sealed\n", encoding="utf-8")
                return "COMPLETE", ("output.txt",), {"input": "hash"}, {"n": 1}

            with patch("scripts.p6.run_c1._sources", return_value={"mock": "fixed"}):
                result = _stage(root, "sample", work)
                self.assertEqual(result["status"], "COMPLETE")
                self.assertEqual(_validate_stage(root, "sample")["details"], {"n": 1})
                with self.assertRaises(FileExistsError):
                    _stage(root, "sample", work)
                (root / "sample/output.txt").write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "output drift"):
                    _validate_stage(root, "sample")


if __name__ == "__main__":
    unittest.main()
