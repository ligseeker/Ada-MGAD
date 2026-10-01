import numpy as np
import pytest

from src.e2e.flat_trigger_features import flatten_trigger_windows


def test_flatten_preserves_each_causal_window_and_canonical_order():
    rng = np.random.RandomState(42)
    arrays = [rng.rand(*shape).astype(np.float32) for shape in
              ((30, 10, 48), (30, 10, 32), (30, 10, 10, 8))]
    indices = np.array([0, 10, 20])
    rows = flatten_trigger_windows(*arrays, indices)
    assert rows.shape == (3, 16000)
    for row, start in zip(rows, indices):
        expected = np.concatenate([array[start:start + 10].reshape(-1) for array in arrays])
        np.testing.assert_array_equal(row, expected)
    changed = [array.copy() for array in arrays]
    for array in changed:
        array[10:] = 100
    np.testing.assert_array_equal(rows[0], flatten_trigger_windows(*changed, indices)[0])
    np.testing.assert_array_equal(rows[::-1], flatten_trigger_windows(*arrays, indices[::-1]))


def test_flatten_rejects_windows_outside_timeline():
    arrays = [np.zeros(shape) for shape in ((20, 10, 48), (20, 10, 32), (20, 10, 10, 8))]
    for indices in ([-1], [11]):
        with pytest.raises(ValueError, match="indices"):
            flatten_trigger_windows(*arrays, indices)
