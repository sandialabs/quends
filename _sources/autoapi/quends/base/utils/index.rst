quends.base.utils
=================

.. py:module:: quends.base.utils


Functions
---------

.. autoapisummary::

   quends.base.utils.power_law_model
   quends.base.utils.to_native_types


Module Contents
---------------

.. py:function:: power_law_model(n, A, p)

.. py:function:: to_native_types(obj)

   Recursively convert NumPy scalar and array types in nested structures to native Python types.

   This function walks through dictionaries, lists, tuples, NumPy scalars, and arrays,
   converting them into Python built-ins:

   - NumPy scalar → Python int or float
   - NumPy array  → Python list (recursively)

   Parameters
   ----------
   obj : any
       The object to convert. Supported container types are dict, list, tuple,
       NumPy ndarray/scalar. Other types are returned unchanged.

   Returns
   -------
   any
       A new object mirroring the input structure but with all NumPy data types replaced
       by their native Python equivalents.


