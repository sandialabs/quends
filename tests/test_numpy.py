import numpy as np
import pandas as pd
import pytest

from quends import DataStream, from_numpy


# Test loading a valid 1D NumPy array
def test_from_numpy_1d():
    np_array = np.array([1, 2, 3, 4, 5])  # 1D array
    data_stream = from_numpy(np_array)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "df"
    ), "DataStream does not have a 'data_frame' attribute."

    # Check the contents of the DataFrame
    df = data_stream.df
    expected_df = pd.DataFrame(np_array, columns=["data"])
    pd.testing.assert_frame_equal(df, expected_df)


# Test loading a valid 2D NumPy array
def test_from_numpy_2d():
    np_array = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
    data_stream = from_numpy(np_array)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "df"
    ), "DataStream does not have a 'data_frame' attribute."

    # Check the contents of the DataFrame
    df = data_stream.df
    expected_df = pd.DataFrame(np_array, columns=["col_0", "col_1", "col_2"])
    pd.testing.assert_frame_equal(df, expected_df)


# Test loading a 2D NumPy array with specific variable names
def test_from_numpy_2d_with_variables():
    np_array = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
    variables = ["A", "B", "C"]
    data_stream = from_numpy(np_array, variables=variables)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "df"
    ), "DataStream does not have a 'data_frame' attribute."

    # Check the contents of the DataFrame
    df = data_stream.df
    expected_df = pd.DataFrame(np_array, columns=variables)
    pd.testing.assert_frame_equal(df, expected_df)


# Test handling of invalid input (not a NumPy array)
def test_from_numpy_invalid_input():
    with pytest.raises(ValueError, match="Input must be a NumPy array."):
        from_numpy("not_a_numpy_array")  # Pass a string instead of a NumPy array


# Test handling of invalid 2D array with mismatched variable names
def test_from_numpy_2d_mismatched_variables():
    np_array = np.array([[1, 2], [3, 4]])  # 2D array
    variables = ["A"]  # Mismatched variable length

    with pytest.raises(
        ValueError,
        match="Length of variables does not match number of columns in the array.",
    ):
        from_numpy(np_array, variables=variables)


# Test handling of invalid array dimensions
def test_from_numpy_invalid_ndim():
    np_array = np.array(42)  # Scalar value

    with pytest.raises(ValueError, match="Only 1D or 2D NumPy arrays are supported."):
        from_numpy(np_array)
