import numpy as np
import pandas as pd
import pytest

from quends import DataStream, from_numpy


def test_from_numpy_1d_loads_named_single_column():
    np_array = np.array([1, 2, 3, 4, 5])
    data_stream = from_numpy(np_array, variable="signal")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data,
        pd.DataFrame({"signal": [1, 2, 3, 4, 5]}),
    )


def test_from_numpy_invalid_input():
    with pytest.raises(ValueError, match="Input must be a NumPy array."):
        from_numpy("not_a_numpy_array", variable="signal")


def test_from_numpy_rejects_2d_arrays():
    np_array = np.array([[1, 2, 3], [4, 5, 6]])

    with pytest.raises(ValueError, match="Only 1D NumPy arrays are supported, got 2D."):
        from_numpy(np_array, variable="signal")


def test_from_numpy_invalid_ndim():
    np_array = np.array(42)

    with pytest.raises(ValueError, match="Only 1D NumPy arrays are supported, got 0D."):
        from_numpy(np_array, variable="signal")
