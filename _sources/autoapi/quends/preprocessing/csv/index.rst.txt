quends.preprocessing.csv
========================

.. py:module:: quends.preprocessing.csv


Functions
---------

.. autoapisummary::

   quends.preprocessing.csv.from_csv


Module Contents
---------------

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


