quends.preprocessing.dictionary
===============================

.. py:module:: quends.preprocessing.dictionary


Functions
---------

.. autoapisummary::

   quends.preprocessing.dictionary.from_dict


Module Contents
---------------

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


