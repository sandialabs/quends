quends.preprocessing.dictionary
===============================

.. py:module:: quends.preprocessing.dictionary


Functions
---------

.. autoapisummary::

   quends.preprocessing.dictionary.from_dict


Module Contents
---------------

.. py:function:: from_dict(data_dict, variables=None)

   Load a data stream from a dictionary.

   Args:
       data_dict (dict): A dictionary where keys are column names and values are lists or arrays of data.
       variables (list, optional): List of variable names (columns) to include.
                                   If None, all dictionary keys are used.

   Returns:
       DataStream: A DataStream object containing the data from the dictionary.


