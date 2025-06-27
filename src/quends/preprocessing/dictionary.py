import pandas as pd

from ..base.data_stream import (  # Adjust import based on your module structure
    DataStream,
)


def from_dict(data_dict, variables=None):
    """
    Load a data stream from a dictionary.

    Args:
        data_dict (dict): A dictionary where keys are column names and values are lists or arrays of data.
        variables (list, optional): List of variable names (columns) to include.
                                    If None, all dictionary keys are used.

    Returns:
        DataStream: A DataStream object containing the data from the dictionary.
    """
    if not isinstance(data_dict, dict):
        raise ValueError("Input must be a dictionary.")

    df = pd.DataFrame(data_dict)

    # If variables is not provided, use all columns.
    if variables is None:
        variables = df.columns.tolist()

    # Filter the DataFrame to include only the specified columns.
    df = df[variables]

    return DataStream(df)
