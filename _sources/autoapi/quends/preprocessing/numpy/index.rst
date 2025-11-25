quends.preprocessing.numpy
==========================

.. py:module:: quends.preprocessing.numpy


Functions
---------

.. autoapisummary::

   quends.preprocessing.numpy.from_numpy


Module Contents
---------------

.. py:function:: from_numpy(np_array, variables=None)

   Load a data stream from a NumPy array.

   Args:
       np_array (np.ndarray): A 1D or 2D NumPy array.
       variables (list, optional): List of column names. For a 1D array, a single-column name is used.
                                   For a 2D array, the length of variables must match the number of columns.
                                   If None, default column names are assigned.

   Returns:
       DataStream: A DataStream object containing the NumPy array data.


