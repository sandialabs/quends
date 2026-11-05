quends.postprocessing.loader
============================

.. py:module:: quends.postprocessing.loader


Classes
-------

.. autoapisummary::

   quends.postprocessing.loader.Loader
   quends.postprocessing.loader.JsonLoader


Module Contents
---------------

.. py:class:: Loader(filepath)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: filepath


   .. py:method:: load(filepath = None)
      :abstractmethod:



.. py:class:: JsonLoader(filepath)

   Bases: :py:obj:`Loader`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: load(filepath = None)

      Load a DataStream from a JSON file, reconstructing its history.



