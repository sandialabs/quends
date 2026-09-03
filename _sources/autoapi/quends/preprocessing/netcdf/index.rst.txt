quends.preprocessing.netcdf
===========================

.. py:module:: quends.preprocessing.netcdf


Functions
---------

.. autoapisummary::

   quends.preprocessing.netcdf.from_netcdf


Module Contents
---------------

.. py:function:: from_netcdf(file, variable)

   Load a single variable from a NetCDF4 file into a data stream.

   Variables that end with ``'_t'`` or ``'_st'`` are extracted from the
   ``Diagnostics`` group and aligned to the ``time`` grid. The returned
   :class:`DataStream` contains the ``time`` column together with the requested
   ``variable`` column.

   :Parameters: * **file** (*str*) -- Path to the NetCDF4 file.
                * **variable** (*str*) -- The variable name to load. Must exist in the file.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If the file does not exist or the variable is not found.


