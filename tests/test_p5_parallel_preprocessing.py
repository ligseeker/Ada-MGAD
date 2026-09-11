import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.e2e.ad_preprocess import build_log_arrays, build_metric_arrays, build_trace_arrays
from src.e2e.gaia_rca_adapter import build_raw_index
from src.e2e.protocol import (
    GAIA_SERVICES, layout_digest, load_config, preprocessing_runtime,
)
from scripts.p5.run_i1_rca_features import materialize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_tiny_raw_root(root: Path) -> dict:
    metric_root = root / "metric/metric_split/metric"
    log_root = root / "business/business_split/business"
    trace_root = root / "trace/trace_split/trace"
    metric_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    trace_root.mkdir(parents=True)
    for service_index, service in enumerate(GAIA_SERVICES):
        pd.DataFrame({
            "timestamp": [1625097600000, 1625097630000],
            "value": [service_index + 1.0, service_index + 2.0],
        }).to_csv(
            metric_root / (
                "{}_0.0.0.1_request_count_2021-07-01_2021-07-15.csv".format(service)
            ),
            index=False,
        )
        pd.DataFrame({
            "message": [
                "2021-07-01 08:00:00,000 | INFO | first",
                "2021-07-01 08:00:30,000 | ERROR | second",
            ],
        }).to_csv(log_root / "business_table_{}_2021-07.csv".format(service), index=False)
        pd.DataFrame({
            "start_time": ["2021-07-01 08:00:00.000000"],
            "end_time": ["2021-07-01 08:00:01.000000"],
            "status_code": [[200, 300, 400, 500][service_index % 4]],
            "service_name": [service],
        }).to_csv(trace_root / "trace_table_{}_2021-07.csv".format(service), index=False)
    return {
        "metrics": layout_digest(metric_root, metric_root.glob("*.csv")),
        "logs": layout_digest(log_root, log_root.glob("*.csv")),
        "traces": layout_digest(trace_root, trace_root.glob("*.csv")),
    }


class ParallelAdPreprocessingTests(unittest.TestCase):
    def test_trace_preprocessing_is_identical_for_one_or_multiple_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_root = root / "trace"
            trace_root.mkdir()
            for service_index, service in enumerate(GAIA_SERVICES):
                parent = GAIA_SERVICES[(service_index - 1) % len(GAIA_SERVICES)]
                pd.DataFrame({
                    "span_id": ["span-{}".format(service)],
                    "parent_id": ["span-{}".format(parent)],
                    "start_time": ["2021-07-01 08:00:00.000000"],
                    "end_time": ["2021-07-01 08:00:01.000000"],
                    "status_code": [[200, 300, 400, 500][service_index % 4]],
                }).to_csv(
                    trace_root / "trace_table_{}_2021-07.csv".format(service), index=False
                )

            grid = np.asarray([1625097600000, 1625097630000], dtype=np.int64)
            slices = {"train": slice(0, 2), "validation": slice(2, 2), "test": slice(2, 2)}
            serial, serial_graph, serial_stats = build_trace_arrays(
                trace_root, root / "serial-span", grid, slices,
                chunk_rows=1, workers=1,
            )
            parallel, parallel_graph, parallel_stats = build_trace_arrays(
                trace_root, root / "parallel-span", grid, slices,
                chunk_rows=1, workers=3,
            )

            np.testing.assert_array_equal(parallel, serial)
            np.testing.assert_array_equal(parallel_graph, serial_graph)
            for key in (
                "raw_rows_scanned",
                "invalid_timestamp_rows",
                "invalid_duration_rows",
                "negative_duration_rows",
                "unknown_status_rows_in_protocol_grid",
                "unmatched_parent_rows_in_protocol_grid",
                "retained_cross_service_rows",
                "graph_edges",
            ):
                self.assertEqual(parallel_stats[key], serial_stats[key])
            self.assertEqual(parallel_stats["parallel_execution"]["trace"]["task_count"], 10)

    def test_metric_preprocessing_is_identical_and_cache_is_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric_root = root / "metric"
            metric_root.mkdir()
            timestamps = [1625097600000, 1625097630000, 1625097660000]
            for service_index, service in enumerate(GAIA_SERVICES):
                pd.DataFrame({
                    "timestamp": timestamps,
                    "value": [service_index + 1.0, service_index + 2.0, service_index + 3.0],
                }).to_csv(
                    metric_root / (
                        "{}_0.0.0.1_request_count_2021-07-01_2021-07-15.csv".format(service)
                    ),
                    index=False,
                )

            grid = np.asarray(timestamps, dtype=np.int64)
            slices = {"train": slice(0, 2), "validation": slice(2, 3), "test": slice(3, 3)}
            serial, serial_stats = build_metric_arrays(
                metric_root, grid, slices, root / "serial-cache", workers=1
            )
            parallel, parallel_stats = build_metric_arrays(
                metric_root, grid, slices, root / "parallel-cache", workers=3
            )

            np.testing.assert_array_equal(parallel, serial)
            self.assertEqual(parallel_stats["feature_names"], serial_stats["feature_names"])
            self.assertEqual(parallel_stats["normalization"], serial_stats["normalization"])
            self.assertEqual(parallel_stats["parallel_execution"]["requested_workers"], 3)
            self.assertEqual(parallel_stats["parallel_execution"]["task_count"], 10)
            self.assertEqual(list((root / "parallel-cache").glob("*.tmp")), [])

    def test_log_preprocessing_is_identical_for_one_or_multiple_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for service_index, service in enumerate(GAIA_SERVICES):
                messages = [
                    "2021-07-01 08:00:00,000 | INFO | first",
                    "2021-07-01 08:00:30,000 | ERROR | second",
                    "invalid prefix | WARNING | ignored",
                ]
                if service_index % 2:
                    messages[1] = "2021-07-01 08:00:30,000 | DEBUG | second"
                pd.DataFrame({"message": messages}).to_csv(
                    root / "business_table_{}_2021-07.csv".format(service), index=False
                )

            grid = np.asarray([1625097600000, 1625097630000], dtype=np.int64)
            slices = {"train": slice(0, 1), "validation": slice(1, 2), "test": slice(2, 2)}
            serial, serial_stats = build_log_arrays(
                root, grid, slices, chunk_rows=1, workers=1
            )
            parallel, parallel_stats = build_log_arrays(
                root, grid, slices, chunk_rows=1, workers=2
            )

            np.testing.assert_array_equal(parallel, serial)
            for key in (
                "raw_rows_scanned",
                "invalid_message_prefix_timestamps",
                "retained_protocol_rows",
                "train_maxima",
            ):
                self.assertEqual(parallel_stats[key], serial_stats[key])
            self.assertEqual(parallel_stats["parallel_execution"]["requested_workers"], 2)
            self.assertEqual(parallel_stats["parallel_execution"]["task_count"], 10)


class ParallelRcaRawIndexTests(unittest.TestCase):
    def test_failed_worker_never_publishes_final_index_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            build_tiny_raw_root(raw_root)
            broken = raw_root / (
                "trace/trace_split/trace/trace_table_dbservice1_2021-07.csv"
            )
            pd.DataFrame({"start_time": ["bad"]}).to_csv(broken, index=False)
            inventory = {
                name: layout_digest(path, path.glob("*.csv"))
                for name, path in {
                    "metrics": raw_root / "metric/metric_split/metric",
                    "logs": raw_root / "business/business_split/business",
                    "traces": raw_root / "trace/trace_split/trace",
                }.items()
            }
            output = root / "broken-index"
            with self.assertRaises(ValueError):
                build_raw_index(raw_root, output, inventory, chunk_rows=1, workers=2)
            self.assertFalse((output / "index_manifest.json").exists())

    def test_raw_index_is_identical_and_manifest_order_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            inventory = build_tiny_raw_root(raw_root)
            serial = build_raw_index(
                raw_root, root / "serial-index", inventory,
                chunk_rows=1, workers=1,
            )
            parallel = build_raw_index(
                raw_root, root / "parallel-index", inventory,
                chunk_rows=1, workers=3,
            )

            self.assertEqual(serial["build_id"], parallel["build_id"])
            self.assertEqual(serial["metric_series"], parallel["metric_series"])
            self.assertEqual(serial["logs"], parallel["logs"])
            self.assertEqual(serial["traces"], parallel["traces"])
            self.assertEqual(
                [record["service"] for record in parallel["metric_series"]],
                list(GAIA_SERVICES),
            )
            self.assertEqual(
                parallel["parallel_execution"]["metric"]["requested_workers"], 3
            )
            self.assertTrue((root / "parallel-index/index_manifest.json").is_file())
            self.assertEqual(list((root / "parallel-index").rglob("*.tmp")), [])


class ParallelRcaMaterializationTests(unittest.TestCase):
    def test_case_shards_are_ordered_equivalent_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            inventory = build_tiny_raw_root(raw_root)
            index_root = root / "index"
            build_raw_index(raw_root, index_root, inventory, chunk_rows=1, workers=1)
            cases = pd.DataFrame([
                {"case_id": "case-{}".format(index), "start_ms": 1625097600000 + index * 1000,
                 "split": "train" if index < 2 else "test"}
                for index in range(4)
            ])
            case_path = root / "cases.csv"
            cases.to_csv(case_path, index=False)
            config = load_config(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml")

            serial = materialize(
                config, index_root, root / "serial-features", root / "serial-artifacts",
                case_path, workers=1, case_chunk_size=1,
            )
            parallel = materialize(
                config, index_root, root / "parallel-features", root / "parallel-artifacts",
                case_path, workers=2, case_chunk_size=1,
            )
            serial_values = np.load(serial["files"]["features"]["path"])
            parallel_values = np.load(parallel["files"]["features"]["path"])
            np.testing.assert_array_equal(parallel_values, serial_values)
            np.testing.assert_array_equal(
                np.load(parallel["files"]["case_ids"]["path"]),
                cases["case_id"].to_numpy(dtype="U32"),
            )
            self.assertEqual(parallel["parallel_execution"]["task_count"], 4)
            self.assertEqual(parallel["parallel_execution"]["requested_workers"], 2)

            resumed = materialize(
                config, index_root, root / "parallel-features", root / "parallel-artifacts",
                case_path, workers=2, case_chunk_size=1,
            )
            self.assertEqual(resumed["shard_cache"]["cache_hits"], 4)
            np.testing.assert_array_equal(
                np.load(resumed["files"]["features"]["path"]), serial_values
            )

    def test_failed_materialization_does_not_replace_previous_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_root = root / "raw"
            inventory = build_tiny_raw_root(raw_root)
            index_root = root / "index"
            build_raw_index(raw_root, index_root, inventory, chunk_rows=1, workers=1)
            case_path = root / "bad-cases.csv"
            pd.DataFrame([
                {"case_id": "bad-case", "start_ms": "not-a-time", "split": "test"},
            ]).to_csv(case_path, index=False)
            feature_root = root / "features"
            feature_root.mkdir()
            old = np.asarray([123.0], dtype=np.float32)
            np.save(feature_root / "z2_features.npy", old, allow_pickle=False)
            config = load_config(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml")

            with self.assertRaises((TypeError, ValueError)):
                materialize(
                    config, index_root, feature_root, root / "artifacts", case_path,
                    workers=2, case_chunk_size=1,
                )
            np.testing.assert_array_equal(np.load(feature_root / "z2_features.npy"), old)


class ParallelRuntimeConfigTests(unittest.TestCase):
    def test_worker_override_is_bounded_by_frozen_cpu_budget(self):
        config = load_config(PROJECT_ROOT / "configs/e2e/gaia_p5_v1.yaml")
        self.assertEqual(preprocessing_runtime(config, "ad", workers=30)["workers"], 30)
        with self.assertRaisesRegex(ValueError, "CPU budget 30"):
            preprocessing_runtime(config, "ad", workers=31)


if __name__ == "__main__":
    unittest.main()
