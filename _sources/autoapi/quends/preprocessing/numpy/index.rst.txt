quends.preprocessing.numpy
==========================

.. py:module:: quends.preprocessing.numpy


Functions
---------

.. autoapisummary::

   quends.preprocessing.numpy.from_numpy


Module Contents
---------------

.. py:function:: from_numpy(np_array, variable, time=None)

   Load a single-variable data stream (with a time column) from a NumPy array.

   To match the other loaders, the returned :class:`DataStream` carries a
   ``time`` column alongside the named ``variable``. Because a NumPy array has
   no intrinsic time axis, the time values come from one of three sources:

   * **1D array + ``time``** — the supplied ``time`` array (same length).
   * **1D array, no ``time``** — a synthesized integer index ``0, 1, 2, ...``.
   * **Nx2 array** — interpreted as two columns; the (strictly increasing)
     column is taken as ``time`` and the other as ``variable``. If neither is
     monotonic, the first column is assumed to be time.

   :Parameters: * **np_array** (*np.ndarray*) -- A 1D array of values, or an Nx2 array
                  ``[time, variable]``.
                * **variable** (*str*) -- The column name to assign to the data values.
                * **time** (*array-like, optional*) -- Explicit time values for the 1D case
                  (must match the length of ``np_array``). Ignored for Nx2 input.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If the input is not a NumPy array of a supported shape, or
       if a supplied ``time`` array has the wrong length.


