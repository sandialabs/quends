quends.base.stationary
======================

.. py:module:: quends.base.stationary


Classes
-------

.. autoapisummary::

   quends.base.stationary.MakeDataStreamStationaryOperation


Module Contents
---------------

.. py:class:: MakeDataStreamStationaryOperation(column, n_pts_orig, *, operate_safe=False, n_pts_min=50, n_pts_frac_min=0.2, drop_fraction=0.2, verbosity=0)

   Bases: :py:obj:`quends.base.operations.DataStreamOperation`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: column


   .. py:attribute:: n_pts_orig


   .. py:attribute:: is_stationary
      :value: None



   .. py:attribute:: operate_safe
      :value: False



   .. py:attribute:: n_pts_min
      :value: 50



   .. py:attribute:: n_pts_frac_min
      :value: 0.2



   .. py:attribute:: drop_fraction
      :value: 0.2



   .. py:attribute:: verbosity
      :value: 0



