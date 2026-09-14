import json
from pathlib import Path
import unittest

from scripts.p5.finalize_v3_manifest import resolve_output_roots


ROOT = Path(__file__).resolve().parents[1]


class V2OutputPathTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
