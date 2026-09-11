"""Small deterministic multiprocessing primitives for P5-I1 preprocessing."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np


Task = TypeVar("Task")
Result = TypeVar("Result")
SUPPORTED_START_METHODS = ("spawn", "forkserver")
_THREADPOOL_LIMITER = None


def _limit_worker_threads() -> None:
    """Prevent each process from creating another CPU-sized BLAS pool."""

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    global _THREADPOOL_LIMITER
    _THREADPOOL_LIMITER = threadpool_limits(limits=1)


def ordered_process_map(
    function: Callable[[Task], Result],
    tasks: Iterable[Task],
    *,
    workers: int = 1,
    start_method: str = "spawn",
    initializer: Optional[Callable[..., None]] = None,
    initargs: Sequence[object] = (),
) -> Tuple[Tuple[Result, ...], Mapping[str, object]]:
    """Run independent tasks while preserving their submitted order.

    A one-worker call executes in-process and is the compatibility reference.
    Multiprocess execution uses ``Executor.map``, whose result order is defined
    by input order rather than completion order.
    """

    task_list = tuple(tasks)
    requested = int(workers)
    if requested < 1:
        raise ValueError("preprocessing workers must be at least one")
    if start_method not in SUPPORTED_START_METHODS:
        raise ValueError(
            "multiprocessing start method must be one of {}".format(
                ", ".join(SUPPORTED_START_METHODS)
            )
        )
    effective = min(requested, len(task_list)) if task_list else 0
    metadata = {
        "requested_workers": requested,
        "effective_workers": effective,
        "task_count": len(task_list),
        "start_method": "serial" if effective <= 1 else start_method,
        "result_order": "submitted task order",
        "worker_native_threads": 1,
    }
    if not task_list:
        return (), metadata
    if effective <= 1:
        if initializer is not None:
            initializer(*tuple(initargs))
        return tuple(function(task) for task in task_list), metadata
    context = multiprocessing.get_context(start_method)
    worker_initializer = _limit_worker_threads if initializer is None else _combined_initializer
    worker_initargs = () if initializer is None else (initializer, tuple(initargs))
    with ProcessPoolExecutor(
        max_workers=effective,
        mp_context=context,
        initializer=worker_initializer,
        initargs=worker_initargs,
    ) as executor:
        return tuple(executor.map(function, task_list, chunksize=1)), metadata


def _combined_initializer(
    initializer: Callable[..., None], initargs: Sequence[object]
) -> None:
    _limit_worker_threads()
    initializer(*tuple(initargs))


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Publish one NumPy array atomically within its destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=destination.name + ".", suffix=".tmp",
            dir=str(destination.parent), delete=False,
        ) as handle:
            temporary_name = handle.name
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(path: Path, data: Mapping[str, object]) -> None:
    """Publish strict JSON atomically within its destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=destination.name + ".", suffix=".tmp",
            dir=str(destination.parent), delete=False, encoding="utf-8",
        ) as handle:
            temporary_name = handle.name
            json.dump(
                data, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
