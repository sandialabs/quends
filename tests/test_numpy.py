# Tests for the single-variable from_numpy loader (carries a time column).
import numpy as np
import pandas as pd
import pytest

from quends import DataStream, from_numpy


def test_from_numpy_1d_synthesizes_time():
    """1D array -> [time, variable] with a synthesized integer time index."""
    ds = from_numpy(np.array([1, 2, 3, 4, 5]), "HeatFlux_st")
    assert isinstance(ds, DataStream)
    df = ds.data
    expected = pd.DataFrame(
        {"time": np.arange(5, dtype=float), "HeatFlux_st": np.arange(1, 6, dtype=float)}
    )
    pd.testing.assert_frame_equal(df, expected)


def test_from_numpy_1d_with_explicit_time():
    ds = from_numpy(np.array([5.0, 6.0, 5.0]), "q", time=[0.0, 0.5, 1.0])
    expected = pd.DataFrame({"time": [0.0, 0.5, 1.0], "q": [5.0, 6.0, 5.0]})
    pd.testing.assert_frame_equal(ds.data, expected)


def test_from_numpy_nx2_resolves_time_column():
    """Nx2 array is interpreted as [time, variable] (monotonic column = time)."""
    ds = from_numpy(np.array([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]]), "HeatFlux_st")
    expected = pd.DataFrame(
        {"time": [0.0, 1.0, 2.0], "HeatFlux_st": [5.0, 6.0, 7.0]}
    )
    pd.testing.assert_frame_equal(ds.data, expected)


def test_from_numpy_time_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length of 'time'"):
        from_numpy(np.array([1.0, 2.0, 3.0]), "q", time=[0.0, 1.0])


def test_from_numpy_invalid_input():
    with pytest.raises(ValueError, match="Input must be a NumPy array."):
        from_numpy("not_a_numpy_array", "q")


def test_from_numpy_scalar_raises():
    with pytest.raises(ValueError, match="expects a 1D array"):
        from_numpy(np.array(42), "q")


def test_from_numpy_wide_2d_raises():
    with pytest.raises(ValueError, match="expects a 1D array"):
        from_numpy(np.ones((4, 3)), "q")
