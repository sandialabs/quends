quends.postprocessing.writer
============================

.. py:module:: quends.postprocessing.writer


Classes
-------

.. autoapisummary::

   quends.postprocessing.writer.Writer
   quends.postprocessing.writer.JsonWriter


Module Contents
---------------

.. py:class:: Writer(filepath)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: filepath


   .. py:method:: save(stream)
      :abstractmethod:



.. py:class:: JsonWriter(filepath, indent = 2)

   Bases: :py:obj:`Writer`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: indent
      :value: 2



   .. py:method:: save(stream)

      Save the DataStream to a JSON file, including its history.



