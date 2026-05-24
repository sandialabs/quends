import os

import pandas as pd

from ..base.data_stream import DataStream


def from_csv(file, variable):
    """
    Load a data stream from a CSV file.

    Args:
        file (str): The path to the CSV file.
        variable (str): The column name to load. Must exist in the CSV file.

    Returns:
        DataStream: A DataStream object containing the single specified column.

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

    return DataStream(df[[variable]])
