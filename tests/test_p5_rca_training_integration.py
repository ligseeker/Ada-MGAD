from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.p5.run_i1_rca import train_oracle
from src.e2e.protocol import GAIA_SERVICES, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SmallRcaTrainingIntegrationTest(unittest.TestCase):
    def test_chronological_train_and_oracle_outputs(self):
        config = load_config(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features_root = root / "features"
            artifacts = root / "artifacts"
            features_root.mkdir()
            rng = np.random.default_rng(11)
            features = rng.normal(size=(6, 10, 68)).astype(np.float32)
            roots = np.arange(6) % 10
            for case, service in enumerate(roots):
                features[case, service, 0] += 2.0
            case_ids = np.asarray(["case-{}".format(i) for i in range(6)])
            splits = np.asarray(["train", "train", "validation", "validation", "test", "test"])
            anchors = np.arange(6, dtype=np.int64) * 1_000_000
            np.save(features_root / "z2_features.npy", features, allow_pickle=False)
            np.save(features_root / "case_ids.npy", case_ids, allow_pickle=False)
            np.save(features_root / "splits.npy", splits, allow_pickle=False)
            np.save(features_root / "anchors_ms.npy", anchors, allow_pickle=False)
            registry_path = root / "rca_case_registry.csv"
            pd.DataFrame([{
                "case_id": case_ids[index],
                "service": GAIA_SERVICES[roots[index]],
                "fault_type": "fault-{}".format(index % 2),
                "split": splits[index],
                "start_ms": anchors[index],
            } for index in range(6)]).to_csv(registry_path, index=False)

            manifest = train_oracle(
                config, features_root, registry_path, root / "model.npz", artifacts
            )
            self.assertEqual(manifest["case_counts"], {
                "train": 2, "validation": 2, "test": 2,
            })
            self.assertTrue((artifacts / "rca_oracle_predictions.csv").is_file())
            self.assertTrue((artifacts / "root_frequency_predictions.csv").is_file())
            self.assertTrue((artifacts / "rca_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
