import os

import pandas as pd

from ._utils import load_single_variable


def from_csv(file, variable):
    """
    Load a single variable as a data stream from a CSV file.

    The returned :class:`DataStream` contains the ``time`` column (when present)
    together with the requested ``variable`` column, so that downstream
    steady-state trimming and ensemble averaging (which require ``time``) keep
    working.

    Args:
        file (str): The path to the CSV file.
        variable (str): The column name to load. Must exist in the CSV file.

    Returns:
        DataStream: A DataStream containing ``[time, variable]`` (or just
        ``[variable]`` if no ``time`` column is present).

    Raises:
        ValueError: If the file does not exist or the column is not found.
    """
    # Check if the file exists
    if not os.path.isfile(file):
        raise ValueError(f"Error: file '{file}' does not exist.")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Check if the specified variable exists in the DataFrame
    if variable not in df.columns:
        raise ValueError(
            f"Error: variable '{variable}' does not exist in file '{file}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep the time column (auto-detected, standardized to "time") alongside the
    # requested variable, recording load provenance in the stream history.
    return load_single_variable(df, variable, source=file, loader="from_csv")
