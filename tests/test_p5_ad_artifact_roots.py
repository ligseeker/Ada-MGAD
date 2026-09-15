import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.p5.run_i1_ad import (
    parse_args,
    resolve_preprocess_artifact_root,
    validate_manifest_config_binding,
)


class AdaArtifactRootTests(unittest.TestCase):
    def test_preprocess_artifact_root_defaults_to_training_artifact_root(self):
        training_root = Path("runs/example/ad")
        self.assertEqual(
            resolve_preprocess_artifact_root(training_root),
            training_root.resolve(),
        )

    def test_cli_accepts_separate_preprocess_artifact_root(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_i1_ad.py",
                "train",
                "--artifact-root",
                "runs/example/ad",
                "--preprocess-artifact-root",
                "data/frozen/ad",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.preprocess_artifact_root, "data/frozen/ad")

    def test_legacy_manifest_allows_only_shared_input_equivalent_config(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/e2e/gaia_p5_v3_preprocessing_v2.json"
        )
        current = json.loads(config_path.read_text(encoding="utf-8"))
        legacy = copy.deepcopy(current)
        legacy["ad"]["early_stopping_metric"] = "train_total_loss"
        legacy["ad"]["primary_checkpoint"] = "best_train_loss.pt"
        legacy["ad"]["auxiliary_checkpoint"] = "best_train_f1.pt"
        legacy["ad_model"]["checkpoint_policy"] = (
            "best_train_loss_primary_best_train_f1_diagnostic"
        )
        legacy_bytes = (json.dumps(legacy, sort_keys=True) + "\n").encode("utf-8")
        manifest = {
            "config_sha256": hashlib.sha256(legacy_bytes).hexdigest(),
            "git_commit": "a" * 40,
        }
        with patch(
            "scripts.p5.run_i1_ad.subprocess.check_output",
            return_value=legacy_bytes,
        ):
            result = validate_manifest_config_binding(manifest, config_path)
        self.assertEqual(result["mode"], "legacy_git_verified_preprocessing_digest")

        legacy["ad"]["grid_seconds"] = 60
        changed_bytes = (json.dumps(legacy, sort_keys=True) + "\n").encode("utf-8")
        manifest["config_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
        with patch(
            "scripts.p5.run_i1_ad.subprocess.check_output",
            return_value=changed_bytes,
        ):
            with self.assertRaisesRegex(ValueError, "preprocessing semantics"):
                validate_manifest_config_binding(manifest, config_path)


if __name__ == "__main__":
    unittest.main()
