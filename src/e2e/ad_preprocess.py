"""Compatibility entry point for the single GAIA V2 preprocessor."""

from pathlib import Path
from typing import Mapping


def build_ad_data(
    config: Mapping[str, object], project_root: Path, data_root: Path,
    artifact_root: Path, chunk_rows: int = 150000,
    raw_root_override: Path = None, workers: int = 1,
    start_method: str = "spawn",
    config_path: Path = None,
):
    """Run the only supported GAIA preprocessing implementation (V2)."""
    preprocessing = config.get("ad_preprocessing", {})
    if preprocessing.get("schema_version") != "gaia_ad_preprocessing_v2":
        raise ValueError("the legacy V3 preprocessing path has been removed; use GAIA V2 config")
    from .gaia_preprocessing.materialize import materialize_ad_inputs

    project_root = Path(project_root).resolve()
    resolved_config = Path(config_path or project_root / "configs/e2e/gaia_p5_v3_preprocessing_v2.json").resolve()
    policy = preprocessing.get("policy_path")
    schema = preprocessing.get("frozen_schema_path")
    if not policy or not schema:
        raise ValueError("V2 config must declare policy_path and frozen_schema_path")
    return materialize_ad_inputs(
        schema_path=(project_root / str(schema)).resolve(),
        config_path=resolved_config,
        raw_root=(Path(raw_root_override) if raw_root_override is not None else Path(str(config["gaia_raw_root"]))),
        data_root=Path(data_root), artifact_root=Path(artifact_root),
        policy_path=(project_root / str(policy)).resolve(),
        runtime={"workers": int(workers), "chunk_rows": int(chunk_rows), "start_method": str(start_method)},
    )


__all__ = ["build_ad_data"]
