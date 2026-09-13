"""Deterministic Train-only Drain3 fitting and frozen log transformation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Mapping, Sequence, Tuple

import jsonpickle
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

LEVELS = ("INFO", "WARNING", "ERROR", "DEBUG", "UNKNOWN")
_LEVEL_RE = re.compile(r"\|\s*(INFO|WARNING|ERROR|DEBUG)\s*\|", re.IGNORECASE)


def message_payload(message: object) -> str:
    text = str(message).strip()
    parts = text.split(" | ", 5)
    return parts[-1].strip() if parts else text


def message_level(message: object) -> str:
    match = _LEVEL_RE.search(str(message))
    return match.group(1).upper() if match else "UNKNOWN"


def parse_message_timestamp_ms(messages: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    parsed = pd.to_datetime(
        messages.astype("string").str.slice(0, 23),
        format="%Y-%m-%d %H:%M:%S,%f",
        errors="coerce",
    )
    localized = parsed.dt.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="NaT")
    valid = localized.notna().to_numpy()
    # ``astype('int64')`` represents NaT as the minimum int; the validity mask
    # ensures those sentinels never reach the grid map.
    values = localized.astype("int64").to_numpy(dtype=np.int64) // 1_000_000
    return values, valid


def _miner_from_config(config_path: Path) -> TemplateMiner:
    config = TemplateMinerConfig()
    config.load(str(config_path))
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def _restore_drain(miner: TemplateMiner, state: str) -> None:
    loaded = jsonpickle.loads(state)
    if len(loaded.id_to_cluster) > 0 and isinstance(next(iter(loaded.id_to_cluster.keys())), str):
        converted = {}
        for key, value in loaded.id_to_cluster.items():
            text = str(key)
            if text.startswith("json://"):
                text = text[len("json://"):]
            try:
                converted[int(text)] = value
            except ValueError:
                converted[key] = value
        loaded.id_to_cluster = converted
    miner.drain.id_to_cluster = loaded.id_to_cluster
    miner.drain.clusters_counter = loaded.clusters_counter
    miner.drain.root_node = loaded.root_node


def fit_train_templates(
    business_dir: Path,
    train_start_ms: int,
    train_end_ms: int,
    *,
    config_path: Path,
    chunk_rows: int,
) -> Mapping[str, object]:
    """Fit one global Drain3 vocabulary in canonical service/file/row order."""

    miner = _miner_from_config(Path(config_path))
    rows_scanned = 0
    train_rows = 0
    invalid_timestamps = 0
    for service in _canonical_services():
        source = Path(business_dir) / "business_table_{}_2021-07.csv".format(service)
        if not source.is_file():
            raise FileNotFoundError(source)
        for chunk in pd.read_csv(
            source, usecols=["message"], chunksize=int(chunk_rows),
            keep_default_na=False, on_bad_lines="error",
        ):
            rows_scanned += len(chunk)
            messages = chunk["message"].astype("string")
            timestamps, valid = parse_message_timestamp_ms(messages)
            invalid_timestamps += int((~valid).sum())
            selected = valid & (timestamps >= int(train_start_ms)) & (timestamps < int(train_end_ms))
            for message in messages.to_numpy(dtype=str)[selected]:
                miner.add_log_message(message_payload(message))
            train_rows += int(selected.sum())
    clusters = sorted(miner.drain.clusters, key=lambda cluster: int(cluster.cluster_id))
    cluster_ids = [int(cluster.cluster_id) for cluster in clusters]
    templates = {str(int(cluster.cluster_id)): cluster.get_template() for cluster in clusters}
    return {
        "state": jsonpickle.dumps(miner.drain, keys=True),
        "cluster_ids": cluster_ids,
        "templates": templates,
        "rows_scanned": int(rows_scanned),
        "train_rows_fitted": int(train_rows),
        "invalid_train_fit_timestamps": int(invalid_timestamps),
        "fit_interval": "[{}, {})".format(int(train_start_ms), int(train_end_ms)),
        "fit_order": "canonical GAIA service order, source row order",
    }


def _canonical_services():
    from .protocol import GAIA_SERVICES
    return GAIA_SERVICES


def transform_service_frozen(
    source_path: Path,
    grid: np.ndarray,
    *,
    chunk_rows: int,
    state: str,
    config_path: Path,
    cluster_ids: Sequence[int],
    train_start_ms: int,
    train_end_ms: int,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    """Map one business-log file with a matcher that cannot create clusters."""

    feature_count = len(cluster_ids) + 1 + len(LEVELS) + 1
    output = np.zeros((len(grid), feature_count), dtype=np.float32)
    miner = _miner_from_config(Path(config_path))
    _restore_drain(miner, state)
    cluster_to_column = {int(cluster_id): index for index, cluster_id in enumerate(cluster_ids)}
    unknown_column = len(cluster_ids)
    level_offset = unknown_column + 1
    total_column = level_offset + len(LEVELS)
    rows_scanned = invalid_timestamps = retained = unseen = 0
    train_rows = test_rows = 0
    grid_ms = 30_000
    for chunk in pd.read_csv(
        source_path, usecols=["message"], chunksize=int(chunk_rows),
        keep_default_na=False, on_bad_lines="error",
    ):
        rows_scanned += len(chunk)
        messages = chunk["message"].astype("string")
        timestamps, valid_time = parse_message_timestamp_ms(messages)
        invalid_timestamps += int((~valid_time).sum())
        aligned = (timestamps // grid_ms) * grid_ms
        positions = np.searchsorted(grid, aligned)
        inside = positions < len(grid)
        exact = np.zeros(len(positions), dtype=bool)
        exact[inside] = grid[positions[inside]] == aligned[inside]
        selected = valid_time & exact
        if not np.any(selected):
            continue
        selected_messages = messages.to_numpy(dtype=str)[selected]
        selected_positions = positions[selected]
        selected_timestamps = timestamps[selected]
        for row_index, message in enumerate(selected_messages):
            result = miner.match(message_payload(message), full_search_strategy="always")
            cluster_column = unknown_column
            if result is not None and int(result.cluster_id) in cluster_to_column:
                cluster_column = cluster_to_column[int(result.cluster_id)]
            else:
                unseen += 1
            position = int(selected_positions[row_index])
            output[position, cluster_column] += 1.0
            level = message_level(message)
            output[position, level_offset + (LEVELS.index(level) if level in LEVELS else len(LEVELS) - 1)] += 1.0
            output[position, total_column] += 1.0
        retained += int(len(selected_messages))
        train_rows += int(((selected_timestamps >= int(train_start_ms)) & (selected_timestamps < int(train_end_ms))).sum())
        test_rows += int((selected_timestamps >= int(train_end_ms)).sum())
    return output, {
        "raw_rows_scanned": int(rows_scanned),
        "invalid_message_prefix_timestamps": int(invalid_timestamps),
        "retained_protocol_rows": int(retained),
        "unseen_template_rows": int(unseen),
        "train_rows_transformed": int(train_rows),
        "test_rows_transformed": int(test_rows),
        "feature_count": int(feature_count),
    }


def schema_features(cluster_ids: Sequence[int]) -> Tuple[str, ...]:
    return tuple(
        ["template_{}".format(int(cluster_id)) for cluster_id in cluster_ids]
        + ["template_UNK"]
        + ["level_{}".format(level) for level in LEVELS]
        + ["log_total"]
    )
