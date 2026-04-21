quends.preprocessing.csv
========================

.. py:module:: quends.preprocessing.csv


Functions
---------

.. autoapisummary::

   quends.preprocessing.csv.from_csv


Module Contents
---------------

.. py:function:: from_csv(file, variable)

   Load a data stream from a CSV file.

   Args:
       file (str): The path to the CSV file.
       variable (str): The column name to load. Must exist in the CSV file.

   Returns:
       DataStream: A DataStream object containing the single specified column.

   Raises:
       ValueError: If the file does not exist or the column is not found.


