import pandas as pd

from ..base.data_stream import (  # Adjust import based on your module structure
    DataStream,
)


def from_dict(data_dict, variable):
    """
    Load a data stream from a dictionary.

    Args:
        data_dict (dict): A dictionary where keys are column names and values are lists or arrays of data.
        variables (list, optional): List of variable names (columns) to include.
                                    If None, all dictionary keys are used.

    Returns:
        DataStream: A DataStream object containing the data from the dictionary.
    """
    # Validate input
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be a dictionary.")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)

    # If variables is not provided, use all columns.
    if variable not in df.columns:
        raise ValueError(
            f"Error: variable '{variable}' does not exist in the dictionary. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Filter the DataFrame to include only the specified columns.
    df = df[[variable]]

    return DataStream(df)
