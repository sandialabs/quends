import json
import os

import pandas as pd

from ..base.data_stream import DataStream


def from_json(file, variable):
    """
    Load a single columnas a data stream from a JSON file.

    Args:
        file (str): The path to the JSON file.
        variable (str): The column name to load. Must exist in the JSON file.

    Returns:
        DataStream: A DataStream object containing the single specified column.
    """
    # Check if the file exists
    if not os.path.isfile(file):
        raise ValueError(f"Error: file {file} does not exist.")

    try:
        # Try to read JSON directly as a DataFrame (works if JSON is an array of records)
        df = pd.read_json(file)
    except ValueError:
        # If that fails, read the JSON as a dictionary and convert to DataFrame
        with open(file, encoding="utf-8") as f:
            payload = json.load(f)
        df = pd.DataFrame(payload["data"])

    # Check if the specified variable exists in the DataFrame
    if variable not in df.columns:
        raise ValueError(
            f"Error: variable '{variable}' does not exist in file '{file}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    return DataStream(df[[variable]])
