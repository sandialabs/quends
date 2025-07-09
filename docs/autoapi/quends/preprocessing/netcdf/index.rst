quends.preprocessing.netcdf
===========================

.. py:module:: quends.preprocessing.netcdf


Functions
---------

.. autoapisummary::

   quends.preprocessing.netcdf.from_netcdf


Module Contents
---------------

.. py:function:: from_netcdf(file, variables=None)

   Load specified variables from a NetCDF4 file into a pandas DataFrame,
   ensuring all variables have the same length, and extracting only variables
   that end with '_t' or '_st' from the Diagnostics group.

   Args:
       file (str): Path to the NetCDF4 file.
       variables (list, optional): List of variable names to include.
                                   If None, load all eligible variables.

   Returns:
       DataStream: A DataStream object containing the data as a pandas DataFrame.


