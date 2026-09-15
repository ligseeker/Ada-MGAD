"""Filesystem layout and state machine for an independent E2E experiment.

The layout deliberately keeps all mutable outputs below ``run_dir``.  The
telemetry and protocol roots are resolved from the experiment configuration,
but this module never creates or writes to those shared roots.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional


_STATUSES = frozenset(
    {"INITIALIZED", "AD_RUNNING", "AD_COMPLETE", "POST_AD_RUNNING", "COMPLETE", "FAILED"}
)
_TRANSITIONS = {
    "INITIALIZED": frozenset({"AD_RUNNING"}),
    "AD_RUNNING": frozenset({"AD_COMPLETE", "FAILED"}),
    "AD_COMPLETE": frozenset({"POST_AD_RUNNING"}),
    "POST_AD_RUNNING": frozenset({"COMPLETE", "FAILED"}),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_is_within(path: Path, parent: Path, *, allow_equal: bool = True) -> bool:
    """Return whether ``path`` is contained by ``parent`` after resolution."""

    try:
        relative = path.relative_to(parent)
    except ValueError:
        return False
    return allow_equal or relative != Path(".")


def _mapping_value(mapping: Mapping[str, object], key: str) -> object:
    value = mapping.get(key)
    return value if value is not None else None


class ExperimentLayout:
    """Resolve one isolated run and expose its input/output paths.

    ``resolve`` is side-effect free.  Use :meth:`create_new` to publish a new
    run; use :meth:`load_existing` before resuming one.  Output path names are
    intentionally fixed so independent callers cannot redirect a run's files
    outside its run directory through configuration.
    """

    def __init__(
        self,
        project_root: Path,
        run_dir: Path,
        *,
        ad_data_root: Path,
        ad_preprocess_artifact_root: Path,
        protocol_root: Path,
        rca_index_root: Path,
        rca_gt_feature_root: Path,
        rca_gt_artifact_root: Path,
        rca_gt_case_registry: Path,
    ) -> None:
        self._project_root = project_root
        self._run_dir = run_dir
        self._ad_data_root = ad_data_root
        self._ad_preprocess_artifact_root = ad_preprocess_artifact_root
        self._protocol_root = protocol_root
        self._rca_index_root = rca_index_root
        self._rca_gt_feature_root = rca_gt_feature_root
        self._rca_gt_artifact_root = rca_gt_artifact_root
        self._rca_gt_case_registry = rca_gt_case_registry

        # All of these are derived only from the canonical run directory.
        self._logs_root = run_dir / "logs"
        self._state_path = run_dir / "run_state.json"
        self._lock_path = run_dir / "run.lock"
        self._ad_artifact_root = run_dir / "ad"
        self._ad_checkpoint_root = run_dir / "checkpoint"
        self._event_root = run_dir / "events"
        self._rca_artifact_root = run_dir / "rca"
        self._rca_detected_artifact_root = self._rca_artifact_root / "detected"
        self._rca_detected_feature_root = self._rca_artifact_root / "detected_features"
        self._rca_detected_case_registry = (
            self._rca_detected_artifact_root / "rca_case_registry_detected.csv"
        )
        self._rca_model_path = self._rca_artifact_root / "conditional_logit.npz"

        for name, path in self._output_paths().items():
            if not _path_is_within(path.resolve(), run_dir, allow_equal=True):
                raise ValueError("{} escapes run directory: {}".format(name, path))
        for name, path in self._shared_input_paths().items():
            if (
                _path_is_within(path, run_dir, allow_equal=True)
                or _path_is_within(run_dir, path, allow_equal=True)
            ):
                raise ValueError("shared input {} overlaps run directory: {}".format(name, path))

    @classmethod
    def resolve(
        cls, project_root: Path, config: Mapping[str, object], run_dir: Path
    ) -> "ExperimentLayout":
        """Resolve a side-effect-free layout from project/config/run paths.

        Relative paths in the configuration and relative ``run_dir`` are
        project-relative.  Absolute run directories are accepted only when
        they are still contained by ``project_root``.
        """

        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        project = Path(project_root).expanduser().resolve()
        requested_run = Path(run_dir).expanduser()
        resolved_run = (project / requested_run if not requested_run.is_absolute() else requested_run).resolve()
        if not _path_is_within(resolved_run, project, allow_equal=False):
            raise ValueError("run_dir must be a non-root path under project_root")

        def path_value(value: object, name: str) -> Path:
            if value is None or not isinstance(value, (str, os.PathLike)):
                raise ValueError("missing or invalid shared input path: {}".format(name))
            path = Path(value).expanduser()
            return (project / path if not path.is_absolute() else path).resolve()

        # New configurations may use ``shared_inputs``; the aliases preserve
        # compatibility with the checked-in V2 config and make the resolver
        # useful to small test/config fixtures without changing their schema.
        sections = []
        for key in ("shared_inputs", "input_paths", "inputs"):
            value = config.get(key)
            if isinstance(value, Mapping):
                sections.append(value)
        for parent_key in ("experiment_layout", "independent_experiment"):
            parent = config.get(parent_key)
            if isinstance(parent, Mapping):
                for key in ("shared_inputs", "input_paths", "inputs"):
                    value = parent.get(key)
                    if isinstance(value, Mapping):
                        sections.append(value)
        ad_paths = config.get("ad_paths")
        if isinstance(ad_paths, Mapping):
            sections.append(ad_paths)
        rca_paths = config.get("rca_paths")
        if isinstance(rca_paths, Mapping):
            sections.append(rca_paths)
        rca = config.get("rca")
        if isinstance(rca, Mapping):
            sections.append(rca)
        sections.append(config)

        def find(*names: str) -> object:
            for section in sections:
                for name in names:
                    value = _mapping_value(section, name)
                    if value is not None:
                        return value
            return None

        return cls(
            project,
            resolved_run,
            ad_data_root=path_value(
                find("ad_data_root", "data_root") or "data/p5/v3_preprocessing_v2/ad",
                "ad_data_root",
            ),
            ad_preprocess_artifact_root=path_value(
                find("ad_preprocess_artifact_root", "preprocess_artifact_root", "artifact_root")
                or "artifacts/p5/v3_preprocessing_v2/ad",
                "ad_preprocess_artifact_root",
            ),
            protocol_root=path_value(
                find("protocol_root", "gt_output_dir") or "artifacts/p5/v3_preprocessing_v2/protocol",
                "protocol_root",
            ),
            rca_index_root=path_value(
                find("rca_index_root", "index_root", "raw_index_root")
                or "data/p5/v3/rca_raw_index",
                "rca_index_root",
            ),
            rca_gt_feature_root=path_value(
                find("rca_gt_feature_root", "gt_feature_root") or "data/p5/v3/rca_features_gt",
                "rca_gt_feature_root",
            ),
            rca_gt_artifact_root=path_value(
                find("rca_gt_artifact_root", "gt_artifact_root") or "artifacts/p5/v3/rca_gt_features",
                "rca_gt_artifact_root",
            ),
            rca_gt_case_registry=path_value(
                find("rca_gt_case_registry", "gt_case_registry")
                or "artifacts/p5/v3/rca/rca_case_registry_gt.csv",
                "rca_gt_case_registry",
            ),
        )

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def logs_root(self) -> Path:
        return self._logs_root

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def ad_artifact_root(self) -> Path:
        return self._ad_artifact_root

    @property
    def ad_checkpoint_root(self) -> Path:
        return self._ad_checkpoint_root

    @property
    def event_root(self) -> Path:
        return self._event_root

    @property
    def rca_detected_case_registry(self) -> Path:
        return self._rca_detected_case_registry

    @property
    def rca_detected_feature_root(self) -> Path:
        return self._rca_detected_feature_root

    @property
    def rca_detected_artifact_root(self) -> Path:
        return self._rca_detected_artifact_root

    @property
    def rca_model_path(self) -> Path:
        return self._rca_model_path

    @property
    def rca_artifact_root(self) -> Path:
        return self._rca_artifact_root

    @property
    def ad_data_root(self) -> Path:
        return self._ad_data_root

    @property
    def ad_preprocess_artifact_root(self) -> Path:
        return self._ad_preprocess_artifact_root

    @property
    def protocol_root(self) -> Path:
        return self._protocol_root

    @property
    def rca_index_root(self) -> Path:
        return self._rca_index_root

    @property
    def rca_gt_feature_root(self) -> Path:
        return self._rca_gt_feature_root

    @property
    def rca_gt_artifact_root(self) -> Path:
        return self._rca_gt_artifact_root

    @property
    def rca_gt_case_registry(self) -> Path:
        return self._rca_gt_case_registry

    def _output_paths(self) -> Dict[str, Path]:
        return {
            "logs_root": self.logs_root,
            "ad_artifact_root": self.ad_artifact_root,
            "ad_checkpoint_root": self.ad_checkpoint_root,
            "event_root": self.event_root,
            "rca_artifact_root": self.rca_artifact_root,
            "rca_detected_artifact_root": self.rca_detected_artifact_root,
            "rca_detected_feature_root": self.rca_detected_feature_root,
            "rca_detected_case_registry": self.rca_detected_case_registry,
            "rca_model_path": self.rca_model_path,
        }

    def _shared_input_paths(self) -> Dict[str, Path]:
        return {
            "ad_data_root": self.ad_data_root,
            "ad_preprocess_artifact_root": self.ad_preprocess_artifact_root,
            "protocol_root": self.protocol_root,
            "rca_index_root": self.rca_index_root,
            "rca_gt_feature_root": self.rca_gt_feature_root,
            "rca_gt_artifact_root": self.rca_gt_artifact_root,
            "rca_gt_case_registry": self.rca_gt_case_registry,
        }

    def _state_identity(self) -> Dict[str, str]:
        return {"project_root": str(self.project_root), "run_dir": str(self.run_dir)}

    def _validate_state(self, state: object) -> Dict[str, Any]:
        if not isinstance(state, dict):
            raise ValueError("run state must be a JSON object")
        if state.get("project_root") != str(self.project_root) or state.get("run_dir") != str(self.run_dir):
            raise ValueError("run state does not match this project/run directory")
        status = state.get("status")
        if status not in _STATUSES:
            raise ValueError("invalid run state status: {}".format(status))
        return state

    @staticmethod
    def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(path.parent),
                prefix=".{}.".format(path.name), suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
            try:
                directory_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Directory fsync is not available on every platform.
                pass
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def create_new(self, metadata: Mapping[str, object]) -> Dict[str, Any]:
        """Create a run and atomically publish its INITIALIZED state."""

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        try:
            json.dumps(dict(metadata))
        except (TypeError, ValueError) as exc:
            raise TypeError("metadata must be JSON serializable") from exc
        if self.run_dir.exists():
            raise FileExistsError("run directory already exists: {}".format(self.run_dir))
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.run_dir.mkdir()
        except FileExistsError:
            raise FileExistsError("run directory already exists: {}".format(self.run_dir))
        for path in (
            self.logs_root,
            self.ad_artifact_root,
            self.ad_checkpoint_root,
            self.event_root,
            self.rca_artifact_root,
            self.rca_detected_artifact_root,
            self.rca_detected_feature_root,
        ):
            path.mkdir(parents=True, exist_ok=False)
        state: Dict[str, Any] = {
            **self._state_identity(),
            "status": "INITIALIZED",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "metadata": dict(metadata),
            "transitions": [],
        }
        self._write_json_atomic(self.state_path, state)
        return state

    def read_state(self) -> Dict[str, Any]:
        if not self.run_dir.is_dir():
            raise FileNotFoundError("run directory does not exist: {}".format(self.run_dir))
        if not self.state_path.is_file():
            raise FileNotFoundError("run state does not exist: {}".format(self.state_path))
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("invalid run state: {}".format(self.state_path)) from exc
        return self._validate_state(state)

    def load_existing(self) -> Dict[str, Any]:
        """Validate and return an existing run's state without modifying it."""

        state = self.read_state()
        for name, path in self._output_paths().items():
            if name.endswith("_root") and not path.is_dir():
                raise ValueError("existing run is missing output directory {}: {}".format(name, path))
        return state

    def transition(
        self, expected: str, target: str, details: Optional[Mapping[str, object]] = None
    ) -> Dict[str, Any]:
        """Atomically move the run through one permitted lifecycle edge."""

        if expected not in _STATUSES or target not in _STATUSES:
            raise ValueError("unknown run state transition: {} -> {}".format(expected, target))
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("transition details must be a mapping")
        state = self.read_state()
        current = state["status"]
        if current != expected:
            raise ValueError("expected state {}, found {}".format(expected, current))
        if target not in _TRANSITIONS.get(current, frozenset()):
            raise ValueError("illegal run state transition: {} -> {}".format(current, target))
        transition: Dict[str, Any] = {
            "from": current, "to": target, "at": _utc_now(),
        }
        if details is not None:
            transition["details"] = dict(details)
            state["details"] = dict(details)
        state["status"] = target
        state["updated_at"] = transition["at"]
        state.setdefault("transitions", []).append(transition)
        try:
            json.dumps(state)
        except (TypeError, ValueError) as exc:
            raise TypeError("transition details must be JSON serializable") from exc
        self._write_json_atomic(self.state_path, state)
        return state

    def as_dict(self) -> Dict[str, str]:
        """Return JSON-friendly path bindings for manifests and diagnostics."""

        values = {
            "project_root": self.project_root,
            "run_dir": self.run_dir,
            "logs_root": self.logs_root,
            "state_path": self.state_path,
            "lock_path": self.lock_path,
            "ad_artifact_root": self.ad_artifact_root,
            "ad_checkpoint_root": self.ad_checkpoint_root,
            "event_root": self.event_root,
            "rca_detected_case_registry": self.rca_detected_case_registry,
            "rca_detected_feature_root": self.rca_detected_feature_root,
            "rca_detected_artifact_root": self.rca_detected_artifact_root,
            "rca_model_path": self.rca_model_path,
            "rca_artifact_root": self.rca_artifact_root,
            "ad_data_root": self.ad_data_root,
            "ad_preprocess_artifact_root": self.ad_preprocess_artifact_root,
            "protocol_root": self.protocol_root,
            "rca_index_root": self.rca_index_root,
            "rca_gt_feature_root": self.rca_gt_feature_root,
            "rca_gt_artifact_root": self.rca_gt_artifact_root,
            "rca_gt_case_registry": self.rca_gt_case_registry,
        }
        return {name: str(path) for name, path in values.items()}


__all__ = ["ExperimentLayout"]
