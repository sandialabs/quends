import os
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from ._utils import load_single_variable


def from_netcdf(file, variable):
    """
    Load a single variable from a NetCDF4 file into a data stream.

    Variables that end with ``'_t'`` or ``'_st'`` are extracted from the
    ``Diagnostics`` group and aligned to the ``time`` grid. The returned
    :class:`DataStream` contains the ``time`` column together with the requested
    ``variable`` column.

    Args:
        file (str): Path to the NetCDF4 file.
        variable (str): The variable name to load. Must exist in the file.

    Returns:
        DataStream: A DataStream containing ``[time, variable]``.

    Raises:
        ValueError: If the file does not exist or the variable is not found.
    """
    # Check if the file exists
    if not os.path.isfile(file):
        raise ValueError(f"Error: file '{file}' does not exist.")

    # Use a context manager to ensure the dataset is closed properly.
    with Dataset(file, "r") as dataset:
        time = dataset["Grids"]["time"][:].flatten()
        max_length = len(time)
        extracted_data = {"time": time}

        diagnostics = dataset["Diagnostics"]
        # Extract variables that end with '_t' or '_st' and ensure they have the
        # same length as 'time'.
        for var_name in diagnostics.variables:
            if var_name.endswith("_t") or var_name.endswith("_st"):
                data = diagnostics[var_name][:]
                # Warn only for the variable the caller actually requested, so
                # reshaping of the requested column is never silent (H/M12).
                _warn = var_name == variable
                if data.ndim == 0:  # Scalar variable
                    if _warn:
                        warnings.warn(
                            f"Variable '{variable}' is scalar; broadcasting it to "
                            f"the time grid (length {max_length}).", stacklevel=2,
                        )
                    extracted_data[var_name] = [data.item()] * max_length
                elif data.ndim == 1:  # 1D variable
                    if len(data) < max_length:
                        if _warn:
                            warnings.warn(
                                f"Variable '{variable}' (length {len(data)}) is "
                                f"shorter than the time grid ({max_length}); "
                                "padding with NaN.", stacklevel=2,
                            )
                        data = np.pad(
                            data, (0, max_length - len(data)), constant_values=np.nan
                        )
                    extracted_data[var_name] = data
                elif data.ndim == 2:  # 2D variable
                    flat_data = data.flatten()
                    if _warn:
                        warnings.warn(
                            f"Variable '{variable}' is 2D {tuple(data.shape)}; "
                            "flattening to 1D and aligning to the time grid.",
                            stacklevel=2,
                        )
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

    # Check if the specified variable exists in the DataFrame
    if variable not in df.columns:
        raise ValueError(
            f"Error: variable '{variable}' does not exist in file '{file}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep the time column (standardized to "time") alongside the variable,
    # recording load provenance in the stream history.
    return load_single_variable(df, variable, source=file, loader="from_netcdf")
