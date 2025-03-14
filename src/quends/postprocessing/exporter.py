import json
import os

import numpy as np
import pandas as pd


class Exporter:
    """
    A class for exporting data/results in various formats: DataFrame, JSON, dictionary, and NumPy array.
    Provides both display (print to console) and save (to file) functions.
    """

    def __init__(self, output_dir="exported_results"):
        """
        Initialize the Exporter.

        Args:
            output_dir (str): Directory to save the exported files.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # --- Conversion Helper Methods ---
    def to_dataframe(self, data):
        """
        Convert input data to a pandas DataFrame.

        Args:
            data: DataFrame, dict, NumPy array, or any structure convertible to DataFrame.

        Returns:
            pd.DataFrame: The converted DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            try:
                return pd.DataFrame(data)
            except Exception as e:
                raise ValueError("Cannot convert data to DataFrame: " + str(e))

    def to_dictionary(self, data):
        """
        Convert input data to a dictionary.

        Args:
            data: dict, DataFrame, or NumPy array.

        Returns:
            dict: The converted dictionary.
        """
        if isinstance(data, dict):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.to_dict(orient="list")
        elif isinstance(data, np.ndarray):
            return {"data": data.tolist()}
        else:
            raise ValueError("Cannot convert data to dictionary.")

    def to_numpy(self, data):
        """
        Convert input data to a NumPy array.

        Args:
            data: np.ndarray, DataFrame, or dict.

        Returns:
            np.ndarray: The converted NumPy array.
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, dict):
            return pd.DataFrame(data).values
        else:
            raise ValueError("Cannot convert data to NumPy array.")

    def to_json(self, data):
        """
        Convert input data to a JSON string.

        Args:
            data: DataFrame, dict, or NumPy array.

        Returns:
            str: The JSON string.
        """
        if isinstance(data, pd.DataFrame):
            d = data.to_dict(orient="list")
        elif isinstance(data, dict):
            d = data
        elif isinstance(data, np.ndarray):
            d = data.tolist()
        else:
            raise ValueError("Cannot convert data to JSON.")
        return json.dumps(d, indent=2)

    # --- Display Functions ---
    def display_dataframe(self, data, head=None):
        """
        Display data as a DataFrame.

        Args:
            data: Data convertible to DataFrame.
            head (int, optional): If provided, only display the first 'head' rows.
        """
        df = self.to_dataframe(data)
        if head is not None:
            return df.head(head)
        else:
            return df

    def display_dictionary(self, data):
        """
        Display data as a dictionary.

        Args:
            data: Data convertible to dictionary.
        """
        d = self.to_dictionary(data)
        print(d)

    def display_numpy(self, data):
        """
        Display data as a NumPy array.

        Args:
            data: Data convertible to a NumPy array.
        """
        arr = self.to_numpy(data)
        print(arr)

    def display_json(self, data):
        """
        Display data as a JSON string.

        Args:
            data: Data convertible to JSON.
        """
        j = self.to_json(data)
        print(j)

    # --- Save Functions ---
    def save_dataframe(self, data, file_name="dataframe.csv"):
        """
        Save data as a CSV file (DataFrame format).

        Args:
            data: Data convertible to DataFrame.
            file_name (str): Name of the file (default: 'dataframe.csv').
        """
        df = self.to_dataframe(data)
        file_path = os.path.join(self.output_dir, file_name)
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")

    def save_dictionary(self, data, file_name="data_dictionary.json"):
        """
        Save data as a JSON file representing a dictionary.

        Args:
            data: Data convertible to a dictionary.
            file_name (str): Name of the file (default: 'data_dictionary.json').
        """
        d = self.to_dictionary(data)
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Dictionary saved to {file_path}")

    def save_numpy(self, data, file_name="data.npy"):
        """
        Save data as a NumPy array file.

        Args:
            data: Data convertible to a NumPy array.
            file_name (str): Name of the file (default: 'data.npy').
        """
        arr = self.to_numpy(data)
        file_path = os.path.join(self.output_dir, file_name)
        np.save(file_path, arr)
        print(f"NumPy array saved to {file_path}")

    def save_json(self, data, file_name="data.json"):
        """
        Save data as a JSON file.

        Args:
            data: Data convertible to JSON.
            file_name (str): Name of the file (default: 'data.json').
        """
        j = self.to_json(data)
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, "w") as f:
            f.write(j)
        print(f"JSON saved to {file_path}")
