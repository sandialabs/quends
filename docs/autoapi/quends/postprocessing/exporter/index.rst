quends.postprocessing.exporter
==============================

.. py:module:: quends.postprocessing.exporter


Classes
-------

.. autoapisummary::

   quends.postprocessing.exporter.Exporter


Module Contents
---------------

.. py:class:: Exporter(output_dir='exported_results')

   A class for exporting data/results in various formats: DataFrame, JSON, dictionary, and NumPy array.
   Provides both display (print to console) and save (to file) functions.
   Includes automatic conversion of NumPy types to native Python types for compatibility.


   .. py:attribute:: output_dir
      :value: 'exported_results'



   .. py:method:: to_native_types(obj)
      :staticmethod:


      Recursively convert NumPy scalar types in dicts/lists/tuples to native Python types.
      Compatible with NumPy 2.x (no `np.float_`, `np.int_`, etc.).



   .. py:method:: to_dataframe(data)

      Convert input data to a pandas DataFrame.

      Args:
          data: DataFrame, dict, NumPy array, or any structure convertible to DataFrame.

      Returns:
          pd.DataFrame: The converted DataFrame.



   .. py:method:: to_dictionary(data)

      Convert input data to a dictionary, and make all types native Python.

      Args:
          data: dict, DataFrame, or NumPy array.

      Returns:
          dict: The converted dictionary (native types).



   .. py:method:: to_numpy(data)

      Convert input data to a NumPy array.

      Args:
          data: np.ndarray, DataFrame, or dict.

      Returns:
          np.ndarray: The converted NumPy array.



   .. py:method:: to_json(data)

      Convert input data to a JSON string (with native Python types).

      Args:
          data: DataFrame, dict, or NumPy array.

      Returns:
          str: The JSON string.



   .. py:method:: display_dataframe(data, head=None)

      Display data as a DataFrame.

      Args:
          data: Data convertible to DataFrame.
          head (int, optional): If provided, only display the first 'head' rows.



   .. py:method:: display_dictionary(data)

      Display data as a dictionary, with all native types.

      Args:
          data: Data convertible to dictionary.



   .. py:method:: display_numpy(data)

      Display data as a NumPy array.

      Args:
          data: Data convertible to a NumPy array.



   .. py:method:: display_json(data)

      Display data as a JSON string, with all native types.

      Args:
          data: Data convertible to JSON.



   .. py:method:: save_dataframe(data, file_name='dataframe.csv')

      Save data as a CSV file (DataFrame format).

      Args:
          data: Data convertible to DataFrame.
          file_name (str): Name of the file (default: 'dataframe.csv').



   .. py:method:: save_dictionary(data, file_name='data_dictionary.json')

      Save data as a JSON file representing a dictionary.

      Args:
          data: Data convertible to a dictionary.
          file_name (str): Name of the file (default: 'data_dictionary.json').



   .. py:method:: save_numpy(data, file_name='data.npy')

      Save data as a NumPy array file.

      Args:
          data: Data convertible to a NumPy array.
          file_name (str): Name of the file (default: 'data.npy').



   .. py:method:: save_json(data, file_name='data.json')

      Save data as a JSON file (with all native types).

      Args:
          data: Data convertible to JSON.
          file_name (str): Name of the file (default: 'data.json').



