quends.base.ensemble_utils
==========================

.. py:module:: quends.base.ensemble_utils

.. autoapi-nested-parse::

   ensemble_utils.py
   -----------------
   Module-level utility functions for ensemble analysis.

   These functions operate on plain lists of :class:`~quends.base.data_stream.DataStream`
   objects, making them reusable from workflow classes, helper scripts, and the
   ``Ensemble`` class itself without introducing circular dependencies.

   Public API
   ----------
   The module exposes the following helpers::

       validate_members         Check that a list of DataStreams is valid.
       validate_column          Check that a column exists in every member.
       get_common_variables     Column names shared by all members.
       resolve_cols             Normalize column_name to a concrete list of strings.
       check_time_steps_uniformity
                                Classify the time-step regularity of each member.
       interpolate_to_common_time
                                Interpolate all members onto a common regular grid.
       direct_average           Average DataStreams that already share the same grid.
       compute_average_ensemble
                                Build one averaged DataStream (auto-interpolates).
       trim_members             Trim each member and return only non-empty results.



Functions
---------

.. autoapisummary::

   quends.base.ensemble_utils.validate_members
   quends.base.ensemble_utils.validate_column
   quends.base.ensemble_utils.get_common_variables
   quends.base.ensemble_utils.resolve_cols
   quends.base.ensemble_utils.check_time_steps_uniformity
   quends.base.ensemble_utils.interpolate_to_common_time
   quends.base.ensemble_utils.direct_average
   quends.base.ensemble_utils.compute_average_ensemble
   quends.base.ensemble_utils.trim_members


Module Contents
---------------

.. py:function:: validate_members(data_streams)

   Raise an informative exception when ``data_streams`` is not a valid
   non-empty list of :class:`~quends.base.data_stream.DataStream` objects.

   :Parameters: **data_streams** (*list of DataStream*)

   :raises ValueError: If the list is empty.
   :raises TypeError: If *data_streams* is not a list, or any element is not a DataStream.


.. py:function:: validate_column(data_streams, column_name)

   Raise an informative exception when *column_name* is missing from any
   ensemble member.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **column_name** (*str*)

   :raises TypeError: If *column_name* is not a string.
   :raises KeyError: If *column_name* is absent from any member.


.. py:function:: get_common_variables(data_streams)

   Return sorted column names shared by every member, excluding ``'time'``.

   :Parameters: **data_streams** (*list of DataStream*)

   :returns: *list of str*


.. py:function:: resolve_cols(data_streams, column_name)

   Normalize *column_name* to a concrete list of strings.

   :Parameters: * **data_streams** (*list of DataStream*) -- Used to enumerate common variables when *column_name* is ``None``.
                * **column_name** (*str, list of str, or None*) -- ``None`` → all common variables; str → single-element list;
                  list → returned as-is.

   :returns: *list of str*


.. py:function:: check_time_steps_uniformity(data_streams, tol = 1e-08, verbose = False)

   Inspect the time-step regularity of each ensemble member.

   For each member, computes diffs of the ``'time'`` column and classifies as:

   ``"AllEqual"``
       All steps identical (within *tol*).
   ``"AllEqualButLast"``
       All steps equal except the last one.
   ``"NotUniform"``
       Multiple distinct step sizes.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **tol** (*float*) -- Absolute tolerance for step-size comparison.
                * **verbose** (*bool*) -- Print per-member diagnostics.

   :returns: *dict* -- ``{"uniform": bool, "majority_step": float, "members": {…}}``


.. py:function:: interpolate_to_common_time(data_streams, method = 'spline', tol = 1e-08, verbose = False)

   Interpolate all ensemble members onto a common, regular time grid.

   The common grid spans ``[min(t_start), max(t_end)]`` across all members
   using the majority time step.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **method** (*{"spline", "linear"}*) -- Interpolation method.
                * **tol** (*float*) -- Tolerance for step-size uniformity check.
                * **verbose** (*bool*) -- Print grid diagnostics.

   :returns: *(new_data_streams, diagnostics)* -- ``new_data_streams`` is a plain :class:`list` of :class:`DataStream`.

   :raises ValueError: If a valid majority step cannot be determined, or if the common grid
       spans zero range.


.. py:function:: direct_average(data_streams, cols = None, min_coverage = 1)

   Average a list of DataStreams with compatible time grids by stacking and
   computing the per-time-point mean.

   :Parameters: * **data_streams** (*list of DataStream*) -- All members must already share (or be compatible with) the same time
                  points.
                * **cols** (*list of str or None*) -- Columns to average.  Defaults to all columns common to every member
                  (excluding ``'time'``).
                * **min_coverage** (*int*) -- Minimum number of non-NaN members required at a time point for the
                  average to be non-NaN.

   :returns: *(averaged_DataStream, meta)*


.. py:function:: compute_average_ensemble(data_streams, interp_method = 'spline', tol = 1e-08, min_coverage = 1, verbose = False)

   Build a single averaged :class:`~quends.base.data_stream.DataStream` from
   ensemble members.

   If all members share the same time grid (detected via
   :func:`check_time_steps_uniformity`), averages directly.
   If grids differ, interpolates all members to a common grid first.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **interp_method** (*{"spline", "linear"}*) -- Interpolation method used when grids differ.
                * **tol** (*float*) -- Tolerance for uniformity check.
                * **min_coverage** (*int*) -- Minimum number of members that must contribute to a time point.
                * **verbose** (*bool*) -- Print diagnostics when interpolation is triggered.

   :returns: *DataStream* -- Single averaged trace.

   :raises ValueError: If *data_streams* is empty.


.. py:function:: trim_members(data_streams, column_name, strategy = None, method = 'std', window_size = 10, start_time = 0.0, threshold = None, robust = True)

   Trim each member in *data_streams* and return only non-empty results.

   Either pass a pre-built *strategy* object (any
   :class:`~quends.base.trim.TrimStrategy` subclass), or specify *method* and
   associated parameters so that a strategy is built internally via
   :func:`~quends.base.trim.build_trim_strategy`.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **column_name** (*str*) -- Column whose steady-state start drives the trim.
                * **strategy** (*TrimStrategy or None*) -- Pre-built trim strategy.  If ``None``, *method* and companions are used.
                * **method** (*str*) -- Trim strategy name: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                  ``"self_consistent"``, or ``"iqr"``.
                * **window_size** (*int*)
                * **start_time** (*float*)
                * **threshold** (*float or None*)
                * **robust** (*bool*)

   :returns: *list of DataStream* -- Non-empty trimmed members (members whose trimmed result was empty are
             silently dropped).

   :raises ValueError: If *method* is unrecognised.


