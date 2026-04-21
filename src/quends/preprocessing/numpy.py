import numpy as np
import pandas as pd

from ..base.data_stream import DataStream


def from_numpy(np_array, variable):
    """
    Load a single-column data stream from a 1D NumPy array.

    Args:
        np_array (np.ndarray): A 1D NumPy array.
        variable (str): The column name to assign to the array data.

    Returns:
        DataStream: A DataStream object containing the single specified column.

    Raises:
        ValueError: If the input is not a NumPy array or is not 1D.
    """
    if not isinstance(np_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    if np_array.ndim != 1:
        raise ValueError(
            f"Only 1D NumPy arrays are supported, got {np_array.ndim}D. "
            f"Select a single column first (e.g. array[:, i]) before passing to from_numpy."
        )

    df = pd.DataFrame(np_array, columns=[variable])

    return DataStream(df)
