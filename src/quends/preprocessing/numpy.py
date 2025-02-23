import numpy as np
import pandas as pd
from ..base.data_stream import DataStream  # Adjust import based on your module structure

def from_numpy(np_array, variables=None):
    """
    Load a data stream from a NumPy array.

    Args:
        np_array (np.ndarray): A 1D or 2D NumPy array.
        variables (list, optional): List of column names. For a 1D array, a single-column name is used.
                                    For a 2D array, the length of variables must match the number of columns.
                                    If None, default column names are assigned.

    Returns:
        DataStream: A DataStream object containing the NumPy array data.
    """
    if not isinstance(np_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    # For 1D arrays, create a single-column DataFrame.
    if np_array.ndim == 1:
        df = pd.DataFrame(np_array, columns=["data"])
    elif np_array.ndim == 2:
        df = pd.DataFrame(np_array)
        # If user provided column names, verify the length matches the array's number of columns.
        if variables is not None:
            if len(variables) != df.shape[1]:
                raise ValueError("Length of variables does not match number of columns in the array.")
            df.columns = variables
        else:
            # Otherwise, assign default names.
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
    else:
        raise ValueError("Only 1D or 2D NumPy arrays are supported.")
    
    return DataStream(df)