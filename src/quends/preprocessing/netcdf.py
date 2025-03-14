import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# from base.data_stream import DataStream
from ..base.data_stream import DataStream  # Adjust base on structure


def from_netcdf(file, variables=None):
    """
    Load specified variables from a NetCDF4 file into a pandas DataFrame,
    ensuring all variables have the same length, and extracting only variables
    that end with '_t' or '_st' from the Diagnostics group.

    Args:
        file (str): Path to the NetCDF4 file.
        variables (list, optional): List of variable names to include.
                                    If None, load all eligible variables.

    Returns:
        DataStream: A DataStream object containing the data as a pandas DataFrame.
    """
    if not os.path.isfile(file):
        raise ValueError(f"Error: file {file} does not exist.")

    # Use a context manager to ensure the dataset is closed properly.
    with Dataset(file, "r") as dataset:
        time = dataset["Grids"]["time"][:].flatten()
        max_length = len(time)
        extracted_data = {"time": time}

        diagnostics = dataset["Diagnostics"]
        for var_name in diagnostics.variables:
            if var_name.endswith("_t") or var_name.endswith("_st"):
                data = diagnostics[var_name][:]
                if data.ndim == 0:  # Scalar variable
                    extracted_data[var_name] = [data.item()] * max_length
                elif data.ndim == 1:  # 1D variable
                    if len(data) < max_length:
                        data = np.pad(
                            data, (0, max_length - len(data)), constant_values=np.nan
                        )
                    extracted_data[var_name] = data
                elif data.ndim == 2:  # 2D variable
                    flat_data = data.flatten()
                    if len(flat_data) < max_length:
                        flat_data = np.pad(
                            flat_data,
                            (0, max_length - len(flat_data)),
                            constant_values=np.nan,
                        )
                    elif len(flat_data) > max_length:
                        flat_data = flat_data[:max_length]
                    extracted_data[var_name] = flat_data

    df = pd.DataFrame(extracted_data)

    # If variables is not specified, use all columns; otherwise filter to those provided.
    if variables is None:
        variables = df.columns.tolist()  # Load all variable names
    else:
        # Optionally, filter out any variable names not in Dataframe
        variables = [var for var in variables if var in df.columns]

    df = df[variables]

    # Return DataStream initialized with the DataFrame
    return DataStream(df)
