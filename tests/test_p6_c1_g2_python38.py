"""C1 G2 path checks must run in the project's Python 3.8 DAG environment."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.p6 import bind_c1_raw_inputs
from scripts.p6 import check_c1_g2_protocol


class C1G2Python38CompatibilityTest(unittest.TestCase):
    def test_containment_rejects_sibling_prefix(self):
        base = Path("/tmp/c1-root")
        self.assertTrue(check_c1_g2_protocol.is_within(base / "run", base))
        self.assertFalse(check_c1_g2_protocol.is_within(Path("/tmp/c1-root-other/run"), base))
        self.assertFalse(bind_c1_raw_inputs.is_within(Path("/tmp/c1-root-other/run"), base))

    def test_repository_binding_uses_supported_path_api(self):
        source = check_c1_g2_protocol.ROOT / "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
        binding = {"path": str(source.relative_to(check_c1_g2_protocol.ROOT)),
                   "bytes": source.stat().st_size,
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        self.assertEqual(check_c1_g2_protocol.checked_binding(binding), source)

    def test_raw_inventory_uses_supported_path_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in bind_c1_raw_inputs.MODALITIES:
                path = root / relative / "part.csv"
                path.parent.mkdir(parents=True)
                path.write_text("time,value\n1,2\n", encoding="utf-8")
            run_table = root / "run/run/run/run_table.csv"
            run_table.parent.mkdir(parents=True)
            run_table.write_text("case_id\nexample\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({"gaia_raw_root": str(root),
                                          "run_table": {"path": str(run_table)}}), encoding="utf-8")
            with patch.object(bind_c1_raw_inputs, "CONFIG", config):
                result = bind_c1_raw_inputs.inventory()
        self.assertEqual(result["file_count"], 4)


if __name__ == "__main__":
    unittest.main()
