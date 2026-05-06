quends.preprocessing.json
=========================

.. py:module:: quends.preprocessing.json


Functions
---------

.. autoapisummary::

   quends.preprocessing.json.from_json


Module Contents
---------------

.. py:function:: from_json(file, variables=None)

   Load a data stream from a JSON file.

   Args:
       file (str): The path to the JSON file.
       variables (list, optional): List of variable names (columns) to load.
                                   If None, all columns are loaded.

   Returns:
       DataStream: A DataStream object containing the data from the JSON file.


