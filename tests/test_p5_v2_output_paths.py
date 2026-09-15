import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts.p5.finalize_v3_manifest import resolve_output_roots


ROOT = Path(__file__).resolve().parents[1]


class V2OutputPathTests(unittest.TestCase):
    def test_finalizer_accepts_run_dir(self):
        from scripts.p5.finalize_v3_manifest import parse_args

        with patch.object(
            sys,
            "argv",
            ["finalize_v3_manifest.py", "--run-dir", "runs/independent"],
        ):
            args = parse_args()
        self.assertEqual(args.run_dir, "runs/independent")

    def test_config_separates_v2_protocol_ad_event_and_historical_rca_root(self):
        config = json.loads(
            (ROOT / "configs/e2e/gaia_p5_v3_preprocessing_v2.json").read_text(
                encoding="utf-8"
            )
        )
        roots = resolve_output_roots(config, artifact_root="artifacts/p5/v3")
        self.assertEqual(roots["protocol"], (ROOT / config["gt_output_dir"]).resolve())
        self.assertEqual(roots["ad"], (ROOT / config["ad_paths"]["artifact_root"]).resolve())
        self.assertEqual(roots["event"], (ROOT / config["ad_paths"]["event_root"]).resolve())
        self.assertNotEqual(roots["protocol"], roots["root"] / "protocol")

    def test_explicit_roots_override_config(self):
        config = {"gt_output_dir": "v2/protocol", "ad_paths": {"artifact_root": "v2/ad"}}
        roots = resolve_output_roots(
            config, artifact_root="legacy",
            protocol_root="explicit/protocol", ad_artifact_root="explicit/ad",
            ad_checkpoint_root="explicit/checkpoint", event_artifact_root="explicit/events",
        )
        self.assertEqual(roots["protocol"], (ROOT / "explicit/protocol").resolve())
        self.assertEqual(roots["ad"], (ROOT / "explicit/ad").resolve())
        self.assertEqual(roots["checkpoint"], (ROOT / "explicit/checkpoint").resolve())
        self.assertEqual(roots["event"], (ROOT / "explicit/events").resolve())

    def test_run_dir_routes_all_mutable_roots_under_run_and_uses_shared_inputs(self):
        from scripts.p5.finalize_v3_manifest import resolve_output_roots

        config = {
            "gt_output_dir": "shared/protocol",
            "ad_paths": {"data_root": "shared/ad-data", "artifact_root": "shared/ad-preprocess"},
        }
        roots = resolve_output_roots(config, artifact_root="legacy", run_dir="runs/independent")
        layout = roots["layout"]
        self.assertEqual(roots["root"], ROOT / "runs/independent/rca")
        self.assertEqual(roots["protocol"], ROOT / "shared/protocol")
        self.assertEqual(roots["ad"], ROOT / "runs/independent/ad")
        self.assertEqual(roots["checkpoint"], ROOT / "runs/independent/checkpoint")
        self.assertEqual(roots["event"], ROOT / "runs/independent/events")
        self.assertEqual(layout.ad_data_root, ROOT / "shared/ad-data")
        self.assertEqual(layout.ad_preprocess_artifact_root, ROOT / "shared/ad-preprocess")
        for name in ("root", "ad", "checkpoint", "event"):
            self.assertTrue(layout.run_dir in roots[name].parents)

    def test_run_dir_rejects_mutable_root_override(self):
        from scripts.p5.finalize_v3_manifest import resolve_output_roots

        with self.assertRaises(ValueError):
            resolve_output_roots(
                {}, artifact_root="legacy", run_dir="runs/independent",
                ad_artifact_root="artifacts/other",
            )


if __name__ == "__main__":
    unittest.main()
