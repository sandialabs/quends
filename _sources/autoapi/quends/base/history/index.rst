quends.base.history
===================

.. py:module:: quends.base.history


Classes
-------

.. autoapisummary::

   quends.base.history.DataStreamHistoryEntry
   quends.base.history.DataStreamHistory


Module Contents
---------------

.. py:class:: DataStreamHistoryEntry

   Immutable record describing a single operation performed on a data stream.


   .. py:attribute:: operation_name
      :type:  str


   .. py:attribute:: parameters
      :type:  Mapping[str, Any]


.. py:class:: DataStreamHistory(entries = None)

   Ordered collection of operations applied to a DataStream.


   .. py:method:: append(entry)

      Append ``entry`` to this history and return self for chaining.



   .. py:method:: entries()

      Expose the ordered sequence of history entries.



