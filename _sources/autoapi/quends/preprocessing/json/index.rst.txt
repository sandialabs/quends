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


