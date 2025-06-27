import os

import pandas as pd

# from base.data_stream import DataStream
from ..base.data_stream import (  # Adjust the import based on the module structure
    DataStream,
)


# Import from a csv file
def from_csv(file, variables=None):
    """
    Load a data stream from a CSV file.

    Args:
        file (str): The path to the CSV file.
        variables (list): Variable names (columns) to load (default: None, which loads all columns).

    Returns:
        DataStream: A DataStream object containing the data from the CSV file.
    """
    # Check if the file exists
    if not os.path.isfile(file):
        raise ValueError(f"Error: file {file} does not exist.")

    df = pd.read_csv(file)

    # If variables is not specified, use all columns; otherwise filter to those provided.
    if variables is None:
        variables = df.columns.tolist()  # Load all variable names
    else:
        # Optionally, filter out any variable names not in Dataframe
        variables = [var for var in variables if var in df.columns]

    df = df[variables]

    # Return DataStream initialized with the DataFrame
    return DataStream(df)
