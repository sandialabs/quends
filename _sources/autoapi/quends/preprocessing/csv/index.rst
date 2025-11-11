quends.preprocessing.csv
========================

.. py:module:: quends.preprocessing.csv


Functions
---------

.. autoapisummary::

   quends.preprocessing.csv.from_csv


Module Contents
---------------

.. py:function:: from_csv(file, variables=None)

   Load a data stream from a CSV file.

   Args:
       file (str): The path to the CSV file.
       variables (list): Variable names (columns) to load (default: None, which loads all columns).

   Returns:
       DataStream: A DataStream object containing the data from the CSV file.


