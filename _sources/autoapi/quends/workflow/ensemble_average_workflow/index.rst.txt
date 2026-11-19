quends.workflow.ensemble_average_workflow
=========================================

.. py:module:: quends.workflow.ensemble_average_workflow

.. autoapi-nested-parse::

   ensemble_average_workflow.py
   ----------------------------
   Ensemble Average Workflow (Technique 0).

   This workflow implements the *average-then-analyze* ensemble technique:

     1. Receive individual ensemble members (an :class:`~quends.base.ensemble.Ensemble`
        or a plain list of :class:`~quends.base.data_stream.DataStream`).
     2. Check whether their time grids differ.
     3. If time grids differ, interpolate all members to a common grid.
     4. Compute one averaged DataStream trace from the (possibly interpolated) members.
     5. Trim the single averaged trace to remove the initial transient.
     6. Compute statistics on the trimmed averaged trace.
     7. Return a structured result dictionary.

   The order is intentionally **different** from the member-wise workflows:
   members are averaged *before* trimming, so the averaged trace is treated as a
   single DataStream for the statistical analysis step.

   All core logic is delegated to:
   - :mod:`quends.base.ensemble_utils` for time-grid and averaging helpers.
   - :class:`~quends.base.trim.TrimDataStreamOperation` for trimming.
   - :meth:`~quends.base.data_stream.DataStream.compute_statistics` for statistics.

   Typical usage
   -------------
   >>> from quends import Ensemble, from_csv
   >>> from quends.workflow import EnsembleAverageWorkflow
   >>>
   >>> members = [from_csv("run_a.csv", "HeatFlux_st"), from_csv("run_b.csv", "HeatFlux_st"), from_csv("run_c.csv", "HeatFlux_st")]
   >>> workflow = EnsembleAverageWorkflow(
   ...     column_name="HeatFlux_st",
   ...     trim_method="std",
   ...     window_size=200,
   ...     start_time=100.0,
   ...     verbosity=1,
   ... )
   >>> result = workflow.run(members)
   >>> print(result["statistics"])



Classes
-------

.. autoapisummary::

   quends.workflow.ensemble_average_workflow.EnsembleAverageWorkflow


Module Contents
---------------

.. py:class:: EnsembleAverageWorkflow(column_name, trim_method = 'std', window_size = 200, start_time = 0.0, threshold = None, robust = True, interp_method = 'spline', tol = 1e-08, min_coverage = 1, ddof = 1, stats_method = 'non-overlapping', stats_window_size = None, confidence_level = 0.95, ci_method = 'normal', verbosity = 0, keep_intermediate = False)

   Ensemble Average workflow.

   Averages all ensemble members into a single trace, then trims and analyses
   that averaged trace.  This is the *Technique-0 / average-then-analyze*
   approach.

   :Parameters: * **column_name** (*str*) -- Column to analyse.
                * **trim_method** (*str*) -- Trim strategy: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                  ``"self_consistent"``, or ``"iqr"``.
                * **window_size** (*int*) -- Block / rolling-window size passed to the trim strategy.
                * **start_time** (*float*) -- Ignore data before this simulation time during trimming.
                * **threshold** (*float or None*) -- Threshold parameter for strategies that need it (e.g.
                  ``"threshold"`` or ``"iqr"``).
                * **robust** (*bool*) -- Use MAD-based (robust) estimates when the trim strategy supports it.
                * **interp_method** (*{"spline", "linear"}*) -- Interpolation method used when time grids differ.
                * **tol** (*float*) -- Tolerance for time-step uniformity check.
                * **min_coverage** (*int*) -- Minimum number of non-NaN members contributing to a time-point in
                  the average.
                * **ddof** (*int*) -- Degrees of freedom for variance/SEM computation.
                * **stats_method** (*{"non-overlapping", "sliding"}*) -- Block method for ``DataStream.compute_statistics``.
                * **stats_window_size** (*int or None*) -- Window size for statistics computation; auto-selected when ``None``.
                * **verbosity** (*int*) -- ``0`` — silent; ``>0`` — print key steps; ``>1`` — also print
                  intermediate diagnostics.
                * **keep_intermediate** (*bool*) -- If ``True``, include the (untrimmed) averaged DataStream in the
                  result under ``"averaged_stream"``.

   .. attribute:: (all parameters stored with ``_`` prefix, e.g. ``_column_name``)




   .. py:method:: run(ensemble_or_members)

      Execute the Ensemble Average workflow.

      :Parameters: **ensemble_or_members** (*Ensemble or list of DataStream*) -- The raw (untrimmed) ensemble members to analyse.

      :returns: *dict* --

                Mapping with the structure::

                    {
                      "workflow":               "ensemble_average",
                      "n_members":              int,
                      "column_name":            str,
                      "interpolation_required": bool,
                      "common_grid":            dict or None,
                      "averaged_stream":        DataStream or None,
                      "trimmed_stream":         DataStream,
                      "statistics":             dict,
                      "trim_history":           list,
                      "metadata":               dict,
                    }

      :raises TypeError: If members are not DataStream instances.
      :raises ValueError: If the member list is empty, the averaged trace is empty after
          trimming, or the column is not present.
      :raises KeyError: If *column_name* is absent from the averaged trace.



