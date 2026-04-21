quends.preprocessing.numpy
==========================

.. py:module:: quends.preprocessing.numpy


Functions
---------

.. autoapisummary::

   quends.preprocessing.numpy.from_numpy


Module Contents
---------------

.. py:function:: from_numpy(np_array, variable)

   Load a single-column data stream from a 1D NumPy array.

   Args:
       np_array (np.ndarray): A 1D NumPy array.
       variable (str): The column name to assign to the array data.

   Returns:
       DataStream: A DataStream object containing the single specified column.

   Raises:
       ValueError: If the input is not a NumPy array or is not 1D.


