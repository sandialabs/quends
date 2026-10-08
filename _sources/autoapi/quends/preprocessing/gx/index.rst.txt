quends.preprocessing.gx
=======================

.. py:module:: quends.preprocessing.gx


Functions
---------

.. autoapisummary::

   quends.preprocessing.gx.from_gx


Module Contents
---------------

.. py:function:: from_gx(file, variable)

   Load a single variable from a GX output file (.nc or .csv).

   :Parameters: * **file** (*str*) -- Path to the GX output file (``.nc`` or ``.csv``).
                * **variable** (*str*) -- The variable name to load (e.g. ``"HeatFlux_st"``).

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If no variable is specified or the file format is unsupported.


