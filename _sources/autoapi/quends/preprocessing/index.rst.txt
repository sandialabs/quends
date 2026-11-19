quends.preprocessing
====================

.. py:module:: quends.preprocessing

.. autoapi-nested-parse::

   Preprocessing loaders for QUENDS.

   Each loader takes a single ``variable`` and returns a ``DataStream`` containing
   ``[time, variable]`` (the time column is auto-detected and standardized).



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/quends/preprocessing/csv/index
   /autoapi/quends/preprocessing/dictionary/index
   /autoapi/quends/preprocessing/gx/index
   /autoapi/quends/preprocessing/json/index
   /autoapi/quends/preprocessing/netcdf/index
   /autoapi/quends/preprocessing/numpy/index


Functions
---------

.. autoapisummary::

   quends.preprocessing.from_csv
   quends.preprocessing.from_dict
   quends.preprocessing.from_gx
   quends.preprocessing.from_json
   quends.preprocessing.from_netcdf
   quends.preprocessing.from_numpy


Package Contents
----------------

.. py:function:: from_csv(file, variable)

   Load a single variable as a data stream from a CSV file.

   The returned :class:`DataStream` contains the ``time`` column (when present)
   together with the requested ``variable`` column, so that downstream
   steady-state trimming and ensemble averaging (which require ``time``) keep
   working.

   :Parameters: * **file** (*str*) -- The path to the CSV file.
                * **variable** (*str*) -- The column name to load. Must exist in the CSV file.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]`` (or just
             ``[variable]`` if no ``time`` column is present).

   :raises ValueError: If the file does not exist or the column is not found.


.. py:function:: from_dict(data_dict, variable)

   Load a single variable as a data stream from a dictionary.

   The returned :class:`DataStream` contains the ``time`` column (when present)
   together with the requested ``variable`` column.

   :Parameters: * **data_dict** (*dict*) -- A dictionary where keys are column names and values are
                  lists or arrays of data.
                * **variable** (*str*) -- The column name to load. Must exist in the dictionary.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]`` (or just
             ``[variable]`` if no ``time`` key is present).

   :raises ValueError: If the input is not a dictionary or the variable is not found.


.. py:function:: from_gx(file, variable)

   Load a single variable from a GX output file (.nc or .csv).

   :Parameters: * **file** (*str*) -- Path to the GX output file (``.nc`` or ``.csv``).
                * **variable** (*str*) -- The variable name to load (e.g. ``"HeatFlux_st"``).

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If no variable is specified or the file format is unsupported.


.. py:function:: from_json(file, variable)

   Load a single variable as a data stream from a JSON file.

   The JSON may be either an array of records or an object with a top-level
   ``"data"`` key holding such an array. The returned :class:`DataStream`
   contains the ``time`` column (when present) together with the requested
   ``variable`` column.

   :Parameters: * **file** (*str*) -- The path to the JSON file.
                * **variable** (*str*) -- The column name to load. Must exist in the JSON file.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]`` (or just
             ``[variable]`` if no ``time`` column is present).

   :raises ValueError: If the file does not exist or the variable is not found.


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


