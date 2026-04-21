quends.preprocessing.json
=========================

.. py:module:: quends.preprocessing.json


Functions
---------

.. autoapisummary::

   quends.preprocessing.json.from_json


Module Contents
---------------

.. py:function:: from_json(file, variable)

   Load a single columnas a data stream from a JSON file.

   Args:
       file (str): The path to the JSON file.
       variable (str): The column name to load. Must exist in the JSON file.

   Returns:
       DataStream: A DataStream object containing the single specified column.


