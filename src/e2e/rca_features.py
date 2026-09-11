"""Parameterized event-relative Ada-RCA-G representation.

The numerical implementation is copied from Ada-RCA-cleanup's frozen
``src/rca/features.py`` at commit ``a2c620922e7c0ab3615d34654d4a3690d1b22c8e``.
The only scientific difference is temporal parameterization: the P5-I1-D
default is W300-B15 (40 bins, 20 pre-event and 20 post-event bins).  The
channel, normalization, aggregation, missingness, and morphology semantics
remain those of the canonical final implementation.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CHANNELS = ("metric", "log", "trace-error", "trace-latency")
ONSET_THRESHOLD = 3.0


@dataclass(frozen=True)
class TemporalSpec:
    """The event-relative temporal grid used by the feature extractor.

    ``window_seconds`` is the context on each side of the anchor event.  The
    remaining fields are explicit rather than derived so that a serialized
    protocol records the complete temporal contract.
    """

    window_seconds: int = 300
    bin_seconds: int = 15
    pre_bins: int = 20
    post_bins: int = 20
    n_bins: int = 40
    onset_sentinel: float = 300.0
    post_position_denominator: float = 19.0

    def __post_init__(self) -> None:
        integer_fields = ("window_seconds", "bin_seconds", "pre_bins", "post_bins", "n_bins")
        for name in integer_fields:
            value = getattr(self, name)
            if int(value) != value or int(value) <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.pre_bins + self.post_bins != self.n_bins:
            raise ValueError("pre_bins + post_bins must equal n_bins")
        if self.window_seconds * 2 != self.n_bins * self.bin_seconds:
            raise ValueError("window_seconds, bin_seconds, and n_bins are inconsistent")
        if not np.isfinite(float(self.onset_sentinel)) or float(self.onset_sentinel) < 0:
            raise ValueError("onset_sentinel must be finite and non-negative")
        if not np.isfinite(float(self.post_position_denominator)) or float(self.post_position_denominator) <= 0:
            raise ValueError("post_position_denominator must be finite and positive")
        if float(self.post_position_denominator) != float(self.post_bins - 1):
            raise ValueError("post_position_denominator must equal post_bins - 1")
        if float(self.onset_sentinel) != float(self.window_seconds):
            raise ValueError("onset_sentinel must equal the post-window horizon")

    @classmethod
    def w300_b15(cls) -> "TemporalSpec":
        return cls()


DEFAULT_TEMPORAL_SPEC = TemporalSpec.w300_b15()
# Compatibility constants for callers that used the frozen fixed-grid names.
N_BINS = DEFAULT_TEMPORAL_SPEC.n_bins
BIN_SECONDS = float(DEFAULT_TEMPORAL_SPEC.bin_seconds)
PRE_BINS = DEFAULT_TEMPORAL_SPEC.pre_bins
POST_BINS = DEFAULT_TEMPORAL_SPEC.post_bins
ONSET_SENTINEL = DEFAULT_TEMPORAL_SPEC.onset_sentinel


@dataclass(frozen=True)
class CaseFeatureSet:
    case_id: str
    candidates: Tuple[str, ...]
    a: np.ndarray
    base: np.ndarray
    z: np.ndarray
    q: np.ndarray
    q_mask: np.ndarray
    morphology_active: np.ndarray
    z2: np.ndarray
    z3: np.ndarray
    spec: TemporalSpec = DEFAULT_TEMPORAL_SPEC


def _service_for_column(column: str, candidates: Sequence[str], channel: str) -> Optional[str]:
    name = str(column)
    if channel.startswith("trace") and name.startswith("frontendservice_"):
        name = "frontend_" + name[len("frontendservice_"):]
    matches = [candidate for candidate in candidates if name.startswith(candidate + "_")]
    return max(matches, key=len) if matches else None


def _binned_indicators(
    path: Path,
    timestamp_column: str,
    candidates: Sequence[str],
    channel: str,
    anchor_time: float,
    spec: TemporalSpec,
) -> Dict[str, np.ndarray]:
    frame = pd.read_csv(path, low_memory=False)
    if timestamp_column not in frame.columns:
        raise ValueError("{} lacks timestamp column {}".format(path, timestamp_column))
    timestamps = pd.to_numeric(frame[timestamp_column], errors="coerce").to_numpy(dtype=float)
    start = float(anchor_time) - float(spec.window_seconds)
    indices = np.floor((timestamps - start) / float(spec.bin_seconds))
    valid_time = np.isfinite(indices) & (indices >= 0) & (indices < spec.n_bins)
    result: Dict[str, np.ndarray] = {}
    for column in frame.columns:
        if column == timestamp_column:
            continue
        if channel == "metric" and str(column).endswith("_latency-50"):
            continue
        service = _service_for_column(str(column), candidates, channel)
        if service is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = valid_time & np.isfinite(values)
        if not np.any(valid):
            continue
        sums = np.bincount(
            indices[valid].astype(np.int64), weights=values[valid], minlength=spec.n_bins
        )
        counts = np.bincount(indices[valid].astype(np.int64), minlength=spec.n_bins)
        binned = np.full(spec.n_bins, np.nan, dtype=float)
        observed = counts > 0
        binned[observed] = sums[observed] / counts[observed]
        result["{}::{}".format(service, column)] = binned
    return result


def _q_by_service(
    indicators: Mapping[str, np.ndarray],
    candidates: Sequence[str],
    spec: TemporalSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.zeros((len(candidates), spec.n_bins), dtype=float)
    q_mask = np.zeros((len(candidates), spec.n_bins), dtype=bool)
    for service_index, service in enumerate(candidates):
        values = [array for key, array in indicators.items() if key.startswith(service + "::")]
        if not values:
            continue
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != spec.n_bins:
            raise ValueError("indicator arrays must have shape (n_indicators, n_bins)")
        centers = np.full(matrix.shape[0], np.nan, dtype=float)
        scales = np.full(matrix.shape[0], np.nan, dtype=float)
        for index, series in enumerate(matrix):
            pre = series[:spec.pre_bins][np.isfinite(series[:spec.pre_bins])]
            if pre.size == 0:
                continue
            center = float(np.median(pre))
            scale = 1.4826 * float(np.median(np.abs(pre - center)))
            if scale < 1e-6:
                scale = float(np.percentile(pre, 75) - np.percentile(pre, 25)) / 1.349
            if scale >= 1e-6 and np.isfinite(scale):
                centers[index] = center
                scales[index] = scale
        valid_indicators = np.isfinite(centers) & np.isfinite(scales)
        if not np.any(valid_indicators):
            continue
        deviations = np.full(matrix.shape, np.nan, dtype=float)
        deviations[valid_indicators] = (
            matrix[valid_indicators] - centers[valid_indicators, None]
        ) / (scales[valid_indicators, None] + 1e-6)
        magnitudes = np.abs(deviations)
        for bin_index in range(spec.n_bins):
            observed = magnitudes[:, bin_index][np.isfinite(magnitudes[:, bin_index])]
            if observed.size:
                q[service_index, bin_index] = float(np.percentile(observed, 90))
                q_mask[service_index, bin_index] = True
    return q, q_mask


def _base_and_morphology(
    q: np.ndarray, q_mask: np.ndarray, spec: TemporalSpec
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_services = q.shape[0]
    base = np.zeros((n_services, len(CHANNELS), 8), dtype=float)
    z = np.zeros((n_services, len(CHANNELS), spec.n_bins), dtype=float)
    masks = q_mask.astype(float)
    active = np.zeros((n_services, len(CHANNELS)), dtype=float)
    z2 = np.zeros((n_services, len(CHANNELS), 9), dtype=float)
    for service_index in range(n_services):
        for channel_index in range(len(CHANNELS)):
            values = q[service_index, channel_index]
            observed = q_mask[service_index, channel_index]
            pre_observed = observed[:spec.pre_bins]
            post_observed = observed[spec.pre_bins:]
            post_values = values[spec.pre_bins:][post_observed]
            pre_values = values[:spec.pre_bins][pre_observed]
            coverage = float(np.count_nonzero(observed)) / float(spec.n_bins)
            available = bool(pre_values.size and post_values.size)
            magnitude = float(np.max(post_values)) if post_values.size else 0.0
            post_mean = float(np.mean(post_values)) if post_values.size else 0.0
            pre_mean = float(np.mean(pre_values)) if pre_values.size else 0.0
            persistence = (
                float(np.mean(post_values >= ONSET_THRESHOLD)) if post_values.size else 0.0
            )
            onset = float(spec.onset_sentinel)
            onset_missing = 1.0
            for index in range(spec.pre_bins, spec.n_bins - 1):
                if observed[index] and observed[index + 1]:
                    if values[index] >= ONSET_THRESHOLD and values[index + 1] >= ONSET_THRESHOLD:
                        onset = float((index - spec.pre_bins) * spec.bin_seconds)
                        onset_missing = 0.0
                        break
            if not available:
                base[service_index, channel_index] = (
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, coverage, 0.0,
                )
                continue
            base[service_index, channel_index] = (
                magnitude, post_mean, post_mean - pre_mean, onset,
                onset_missing, persistence, coverage, 1.0,
            )
            maximum = float(np.max(values[observed])) if np.any(observed) else 0.0
            if maximum < 1e-6 or not np.isfinite(maximum):
                continue
            active[service_index, channel_index] = 1.0
            normalized = values / (maximum + 1e-6)
            z[service_index, channel_index, observed] = normalized[observed]
            post_indices = np.flatnonzero(post_observed)
            post_z = normalized[spec.pre_bins:][post_observed]
            pre_z = normalized[:spec.pre_bins][pre_observed]
            pre_z_mean = float(np.mean(pre_z)) if pre_z.size else 0.0
            post_z_mean = float(np.mean(post_z)) if post_z.size else 0.0
            peak_position = int(post_indices[post_z.argmax()]) if post_z.size else int(spec.post_bins - 1)
            peak_fraction = (
                float(peak_position) / float(spec.post_position_denominator)
                if post_z.size else 1.0
            )
            weights = np.maximum(post_z, 0.0)
            centroid = (
                float(np.average(post_indices, weights=weights))
                / float(spec.post_position_denominator)
                if np.sum(weights) > 0 else 0.0
            )
            x = post_indices.astype(float) / float(spec.post_position_denominator)
            slope = (
                float(np.polyfit(x, post_z, 1)[0])
                if post_z.size >= 2 and np.ptp(x) > 0 else 0.0
            )
            adjacent = np.abs(np.diff(post_z))[np.diff(post_indices) == 1]
            mean_adjacent = float(np.mean(adjacent)) if adjacent.size else 0.0
            fraction_high = float(np.mean(post_z >= 0.5)) if post_z.size else 0.0
            z2[service_index, channel_index] = (
                pre_z_mean, post_z_mean, post_z_mean - pre_z_mean,
                peak_fraction, centroid, slope, mean_adjacent, fraction_high, 1.0,
            )
    return base, z, masks, active, z2


def _flatten_indicator_channel(
    channel_values: Mapping[str, object], candidates: Sequence[str], spec: TemporalSpec
) -> Dict[str, np.ndarray]:
    """Normalize service->matrix and flat key->series channel mappings."""
    flattened: Dict[str, np.ndarray] = {}
    candidate_set = set(candidates)
    for key, raw in channel_values.items():
        name = str(key)
        if "::" in name:
            service = name.split("::", 1)[0]
            arrays = [raw]
            keys = [name]
        elif name in candidate_set:
            if isinstance(raw, Mapping):
                arrays = list(raw.values())
                keys = ["{}::{}".format(name, indicator) for indicator in raw]
            else:
                matrix = np.asarray(raw, dtype=float)
                if matrix.ndim == 1:
                    arrays = [matrix]
                elif matrix.ndim == 2:
                    arrays = list(matrix)
                else:
                    raise ValueError("service indicator arrays must be 1D or 2D")
                keys = ["{}::{}".format(name, index) for index in range(len(arrays))]
            service = name
        else:
            raise ValueError("indicator key must be '<service>::<indicator>' or a candidate service")
        if service not in candidate_set:
            continue
        for output_key, array in zip(keys, arrays):
            values = np.asarray(array, dtype=float)
            if values.ndim != 1 or values.shape[0] != spec.n_bins:
                raise ValueError("indicator arrays must have length n_bins={}".format(spec.n_bins))
            flattened[output_key] = values
    return flattened


def extract_case_features_from_indicators(
    case_id: str,
    candidates: Sequence[str],
    indicators_by_channel: Mapping[str, Mapping[str, object]],
    spec: TemporalSpec = DEFAULT_TEMPORAL_SPEC,
) -> CaseFeatureSet:
    """Build features from already aligned raw indicator series.

    ``indicators_by_channel`` may use either ``channel -> service -> array``
    (one or more indicators per service, with shape ``(n_indicators, n_bins)``)
    or ``channel -> {"service::indicator": series}``.  Series are raw finite or
    missing values; robust pre-event normalization and Q90 aggregation are
    performed here.
    """
    candidates = tuple(candidates)
    q_channels = []
    mask_channels = []
    for channel in CHANNELS:
        raw_channel = indicators_by_channel.get(channel, {})
        if not isinstance(raw_channel, Mapping):
            raise ValueError("each channel indicator collection must be a mapping")
        indicators = _flatten_indicator_channel(raw_channel, candidates, spec)
        q, mask = _q_by_service(indicators, candidates, spec)
        q_channels.append(q)
        mask_channels.append(mask)
    q = np.stack(q_channels, axis=1)
    q_mask = np.stack(mask_channels, axis=1)
    base, z, masks, active, z2 = _base_and_morphology(q, q_mask, spec)
    available = base[:, :, 7] > 0
    clipped = np.minimum(base[:, :, 0], 20.0)
    a = np.divide(
        clipped.sum(axis=1), available.sum(axis=1),
        out=np.zeros(len(candidates), dtype=float),
        where=available.sum(axis=1) > 0,
    )
    z3 = np.concatenate((base, z, masks, active[:, :, None]), axis=2)
    arrays = (a, base, z, q, masks, active, z2, z3)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("feature extraction produced non-finite values")
    return CaseFeatureSet(
        case_id=case_id,
        candidates=candidates,
        a=a,
        base=base,
        z=z,
        q=q,
        q_mask=masks,
        morphology_active=active,
        z2=z2,
        z3=z3,
        spec=spec,
    )


def extract_case_features(
    case_id: str,
    candidates: Sequence[str],
    anchor_time: float,
    source: object,
    spec: TemporalSpec = DEFAULT_TEMPORAL_SPEC,
) -> CaseFeatureSet:
    """CSV convenience entry point; it delegates to the mapping entry point."""
    paths = {
        "metric": (Path(source.simple_metrics_path), "time"),
        "log": (Path(source.logts_path), "time"),
        "trace-error": (Path(source.trace_error_path), "time"),
        "trace-latency": (Path(source.trace_latency_path), "time"),
    }
    indicators_by_channel = {
        channel: _binned_indicators(path, timestamp, candidates, channel, anchor_time, spec)
        for channel, (path, timestamp) in paths.items()
    }
    return extract_case_features_from_indicators(case_id, candidates, indicators_by_channel, spec)


def flatten_features(features: CaseFeatureSet, variant: str) -> np.ndarray:
    if variant == "z0":
        return features.a[:, None]
    if variant == "z1":
        return features.base.reshape(len(features.candidates), -1)
    if variant == "z2":
        return np.concatenate((features.base, features.z2), axis=2).reshape(len(features.candidates), -1)
    if variant == "z3":
        return features.z3.reshape(len(features.candidates), -1)
    raise ValueError("unknown feature variant {!r}".format(variant))


def feature_health(feature_sets: Sequence[CaseFeatureSet]) -> Mapping[str, object]:
    """Implementation sanity statistics only; never a representation selector."""

    if not feature_sets:
        raise ValueError("feature health requires at least one case")
    rows = np.concatenate([flatten_features(value, "z2") for value in feature_sets], axis=0)
    available = np.concatenate([value.base[:, :, 7] for value in feature_sets], axis=0)
    masks = np.concatenate([value.q_mask for value in feature_sets], axis=0)
    active = np.concatenate([value.morphology_active for value in feature_sets], axis=0)
    finite = np.isfinite(rows)
    all_zero = np.all(np.isclose(rows, 0.0), axis=1)
    all_masked = ~np.any(masks.astype(bool), axis=(1, 2))
    finite_columns = np.all(finite, axis=0)
    constant_columns = np.zeros(rows.shape[1], dtype=bool)
    constant_columns[finite_columns] = np.ptp(rows[:, finite_columns], axis=0) <= 1e-12
    signatures = [hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).digest() for row in rows]
    unique_signatures = len(set(signatures))
    return {
        "purpose": "implementation sanity check only; no performance or representation selection",
        "cases": int(len(feature_sets)),
        "candidate_rows": int(len(rows)),
        "feature_dimension": int(rows.shape[1]),
        "finite_ratio": float(finite.mean()),
        "channel_available_ratio": float(available.mean()),
        "mask_coverage": float(masks.mean()),
        "morphology_active_ratio": float(active.mean()),
        "all_zero_rows": int(all_zero.sum()),
        "all_zero_row_ratio": float(all_zero.mean()),
        "all_masked_rows": int(all_masked.sum()),
        "all_masked_row_ratio": float(all_masked.mean()),
        "constant_features": int(constant_columns.sum()),
        "constant_feature_ratio": float(constant_columns.mean()),
        "unique_candidate_signatures": int(unique_signatures),
        "candidate_signature_duplication": int(len(rows) - unique_signatures),
        "candidate_signature_duplication_ratio": float((len(rows) - unique_signatures) / len(rows)),
    }


def deterministic_shuffle(features: CaseFeatureSet) -> CaseFeatureSet:
    seed_material = "Ada-RCA|P3-G1|20260826|{}".format(features.case_id).encode("utf-8")
    seed = int.from_bytes(__import__("hashlib").sha256(seed_material).digest()[:8], "big")
    rng = np.random.RandomState(seed % (2 ** 32 - 1))
    permutation = rng.permutation(features.spec.n_bins)
    z = features.z[:, :, permutation]
    masks = features.q_mask[:, :, permutation]
    z3 = np.concatenate((features.base, z, masks, features.morphology_active[:, :, None]), axis=2)
    return CaseFeatureSet(
        features.case_id, features.candidates, features.a, features.base, z,
        features.q, masks, features.morphology_active, features.z2, z3, features.spec,
    )
