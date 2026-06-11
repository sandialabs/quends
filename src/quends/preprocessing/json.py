import json
import os

import pandas as pd

from ._utils import load_single_variable


def from_json(file, variable):
    """
    Load a single variable as a data stream from a JSON file.

    The JSON may be either an array of records or an object with a top-level
    ``"data"`` key holding such an array. The returned :class:`DataStream`
    contains the ``time`` column (when present) together with the requested
    ``variable`` column.

    Args:
        file (str): The path to the JSON file.
        variable (str): The column name to load. Must exist in the JSON file.

    Returns:
        DataStream: A DataStream containing ``[time, variable]`` (or just
        ``[variable]`` if no ``time`` column is present).

    Raises:
        ValueError: If the file does not exist or the variable is not found.
    """
    # Check if the file exists
    if not os.path.isfile(file):
        raise ValueError(f"Error: file '{file}' does not exist.")

    try:
        # Try to read JSON directly as a DataFrame (works if JSON is an array of records)
        df = pd.read_json(file)
    except ValueError:
        # Otherwise, load the JSON manually and convert to a DataFrame, accepting
        # either a bare record array or an object with a top-level "data" key.
        with open(file, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        df = pd.DataFrame(payload)

    # Check if the specified variable exists in the DataFrame
    if variable not in df.columns:
        raise ValueError(
            f"Error: variable '{variable}' does not exist in file '{file}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep the time column (auto-detected, standardized to "time") alongside the
    # requested variable, recording load provenance in the stream history.
    return load_single_variable(df, variable, source=file, loader="from_json")
