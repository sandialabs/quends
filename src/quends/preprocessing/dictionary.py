import pandas as pd

from ._utils import load_single_variable


def from_dict(data_dict, variable):
    """
    Load a single variable as a data stream from a dictionary.

    The returned :class:`DataStream` contains the ``time`` column (when present)
    together with the requested ``variable`` column.

    Args:
        data_dict (dict): A dictionary where keys are column names and values are
            lists or arrays of data.
        variable (str): The column name to load. Must exist in the dictionary.

    Returns:
        DataStream: A DataStream containing ``[time, variable]`` (or just
        ``[variable]`` if no ``time`` key is present).

    Raises:
        ValueError: If the input is not a dictionary or the variable is not found.
    """
    # Validate input
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be a dictionary.")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # Check if the specified variable exists in the DataFrame
    if variable not in df.columns:
        raise ValueError(
            f"Error: variable '{variable}' does not exist in the dictionary. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Keep the time column (auto-detected, standardized to "time") alongside the
    # requested variable, recording load provenance in the stream history.
    return load_single_variable(df, variable, source="<dict>", loader="from_dict")
