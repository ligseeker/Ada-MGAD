import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.features import (
    EventCheckpointError,
    load_event_checkpoint,
    write_event_checkpoint,
)


class EventCheckpointTest(unittest.TestCase):
    def test_round_trip_is_deterministic_and_source_bound(self):
        pairs = (("c1", "s1"), ("c2", "s1"))
        values = np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
        observed = np.asarray([[True, False], [True, True]])
        source = {"path": "/raw/input.csv", "bytes": 123}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "checkpoint"
            first = write_event_checkpoint(
                str(output),
                "gaia/log/s1",
                "p2_log_l0_v1",
                ("a", "b"),
                pairs,
                values,
                observed,
                source,
                {"rows": 9},
            )
            first_bytes = {
                name: (output / name).read_bytes()
                for name in (
                    "index.jsonl",
                    "values.npy",
                    "observed.npy",
                    "manifest.json",
                )
            }
            second = write_event_checkpoint(
                str(output),
                "gaia/log/s1",
                "p2_log_l0_v1",
                ("a", "b"),
                pairs,
                values,
                observed,
                source,
                {"rows": 9},
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first_bytes,
                {name: (output / name).read_bytes() for name in first_bytes},
            )
            loaded_values, loaded_observed, manifest = load_event_checkpoint(
                str(output),
                "gaia/log/s1",
                "p2_log_l0_v1",
                ("a", "b"),
                pairs,
                source,
            )
            np.testing.assert_array_equal(loaded_values, values)
            np.testing.assert_array_equal(loaded_observed, observed)
            self.assertEqual(manifest["extraction_stats"]["rows"], 9)
            with self.assertRaises(EventCheckpointError):
                load_event_checkpoint(
                    str(output),
                    "gaia/log/s1",
                    "p2_log_l0_v1",
                    ("a", "b"),
                    pairs,
                    {"path": "/raw/input.csv", "bytes": 124},
                )

    def test_rejects_tampering_and_nonzero_masked_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "checkpoint"
            with self.assertRaises(EventCheckpointError):
                write_event_checkpoint(
                    str(output),
                    "id",
                    "extractor",
                    ("a",),
                    (("c", "s"),),
                    [[1.0]],
                    [[False]],
                    {},
                    {},
                )
            write_event_checkpoint(
                str(output),
                "id",
                "extractor",
                ("a",),
                (("c", "s"),),
                [[1.0]],
                [[True]],
                {},
                {},
            )
            with (output / "values.npy").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(EventCheckpointError):
                load_event_checkpoint(
                    str(output),
                    "id",
                    "extractor",
                    ("a",),
                    (("c", "s"),),
                    {},
                )


if __name__ == "__main__":
    unittest.main()
