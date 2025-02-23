import os
import json
import pandas as pd
from ..base.data_stream import DataStream

def from_json(file, variables=None):
    """
    Load a data stream from a JSON file.

    Args:
        file (str): The path to the JSON file.
        variables (list, optional): List of variable names (columns) to load.
                                    If None, all columns are loaded.

    Returns:
        DataStream: A DataStream object containing the data from the JSON file.
    """
    if not os.path.isfile(file):
        raise ValueError(f"Error: file {file} does not exist.")
    
    try:
        # Try to read JSON directly as a DataFrame (works if JSON is an array of records)
        df = pd.read_json(file)
    except ValueError:
        # Otherwise, load JSON data manually and convert to DataFrame
        with open(file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    
    if variables is None:
        variables = df.columns.to_list()
    else:
        # Optionally, filter out any variable names not in Dataframe
        variables = [var for var in variables if var in df.columns]
    
    df = df[variables]

    # Return DataStream initialized with the DataFrame
    return DataStream(df)