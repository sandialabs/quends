quends.workflow
===============

.. py:module:: quends.workflow


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/quends/workflow/batch_ensemble_workflow/index
   /autoapi/quends/workflow/ensemble_average_workflow/index
   /autoapi/quends/workflow/ensemble_statistics_workflow/index
   /autoapi/quends/workflow/robust_workflow/index


Classes
-------

.. autoapisummary::

   quends.workflow.BatchEnsembleWorkflow
   quends.workflow.EnsembleAverageWorkflow
   quends.workflow.EnsembleStatisticsWorkflow
   quends.workflow.RobustWorkflow


Package Contents
----------------

.. py:class:: BatchEnsembleWorkflow(ensemble_groups = None, column_name = None, sub_workflow_type = 'statistics', technique = 'both', batch_config = None, verbosity = 0)

   Batch Ensemble workflow — apply a sub-workflow to each ensemble group.

   ``run(groups)`` dispatches every group to the configured sub-workflow and
   returns a structured dict ``{n_items, n_success, results, errors, metadata}``.
   (Cross-group aggregation via :meth:`_aggregate_results` is not yet implemented.)

   :Parameters: * **ensemble_groups** (*list or None*) -- A list of ensemble groups to process.  Each group should be an
                  :class:`~quends.base.ensemble.Ensemble` object or a list of
                  :class:`~quends.base.data_stream.DataStream` objects.
                  Validated at construction time; ``None`` is accepted (groups can be
                  passed later to :meth:`run`).
                * **column_name** (*str or None*) -- Column to analyse in each ensemble group.
                * **sub_workflow_type** (*{"average", "statistics"}*) -- Which sub-workflow to apply to each ensemble group.
                  ``"average"`` → :class:`~quends.workflow.EnsembleAverageWorkflow`;
                  ``"statistics"`` → :class:`~quends.workflow.EnsembleStatisticsWorkflow`.
                * **technique** (*{"technique1", "technique2", "both"}*) -- Technique to use when *sub_workflow_type* is ``"statistics"``.
                * **batch_config** (*dict or None*) -- Arbitrary configuration dict forwarded to the sub-workflow constructor.
                  See the sub-workflow class docs for valid keys.
                * **verbosity** (*int*) -- ``0`` — silent; ``>0`` — print progress.

   :ivar _ensemble_groups:
   :vartype _ensemble_groups: list or None
   :ivar _column_name:
   :vartype _column_name: str or None
   :ivar _sub_workflow_type:
   :vartype _sub_workflow_type: str
   :ivar _technique:
   :vartype _technique: str
   :ivar _batch_config:
   :vartype _batch_config: dict
   :ivar _verbosity:
   :vartype _verbosity: int


   .. py:property:: n_groups
      :type: Optional[int]


      Number of ensemble groups, or ``None`` if not yet provided.


   .. py:property:: sub_workflow_type
      :type: str



   .. py:property:: technique
      :type: str



   .. py:method:: run(ensemble_groups = None, continue_on_error = True)

      Execute the batch workflow over all ensemble groups.

      Each group is dispatched to the appropriate sub-workflow
      (:class:`~quends.workflow.EnsembleAverageWorkflow` for ``"average"``,
      :class:`~quends.workflow.EnsembleStatisticsWorkflow` for
      ``"statistics"``).  This is an in-process, object-level orchestrator —
      file discovery and CSV/NetCDF loading are explicitly out of scope and
      remain future work.

      Each batch item may be:

      * an :class:`~quends.base.ensemble.Ensemble` instance,
      * a list of :class:`~quends.base.data_stream.DataStream` members,
      * a ``dict`` with keys:

        - ``"members"`` *or* ``"ensemble"`` — required, an Ensemble or list of
          DataStreams,
        - ``"name"`` or ``"id"`` — optional human-readable identifier,
        - ``"workflow_type"`` — optional per-item override (``"average"`` or
          ``"statistics"``),
        - ``"workflow_config"`` — optional per-item config dict merged on top
          of ``self._batch_config``.

      :Parameters: * **ensemble_groups** (*list or None*) -- Override the groups provided at construction time.
                   * **continue_on_error** (*bool*) -- If ``True`` (default), catch per-item exceptions and record them
                     in the ``"errors"`` map; if ``False``, re-raise the first failure.

      :returns: *dict* --

                Mapping with the structure::

                    {"workflow": "batch_ensemble", "n_items": int, "n_success": int,
                     "n_failed": int, "results": {id: per-item result}, "errors":
                     {id: error message}, "metadata": {...}}

      :raises ValueError: If no groups are available, or if a per-item dict is malformed.
      :raises TypeError: If a batch item is not an Ensemble, list, or dict.

      .. rubric:: Notes

      File-based batch loading (CSV directories, NetCDF globs, …) is future
      work; this method only orchestrates already-constructed objects.



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

   :ivar (all parameters stored with ``_`` prefix, e.g. ``_column_name``):


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



.. py:class:: EnsembleStatisticsWorkflow(column_name = None, technique = 'both', trim_method = 'std', window_size = 200, start_time = 0.0, threshold = None, robust = True, trim_strategy = None, ddof = 1, stats_method = 'non-overlapping', stats_window_size = None, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal', verbosity = 0, keep_trimmed = False)

   Ensemble Statistics workflow supporting Technique 1 and/or Technique 2.

   :Parameters: * **column_name** (*str or None*) -- Column to analyse.  ``None`` runs all common variables.
                * **technique** (*{"technique1", "technique2", "both"}*) -- Which technique(s) to run.
                * **trim_method** (*str*) -- Trim strategy applied to each member before analysis:
                  ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                  ``"self_consistent"``, or ``"iqr"``.
                * **window_size** (*int*) -- Block / rolling-window size for the trim strategy.
                * **start_time** (*float*) -- Ignore data before this simulation time during trimming.
                * **threshold** (*float or None*) -- Threshold parameter for trim strategies that need it.
                * **robust** (*bool*) -- Use MAD-based (robust) estimates in the trim strategy where
                  supported.
                * **trim_strategy** (*TrimStrategy or None*) -- Pre-built trim strategy object.  When provided, *trim_method*,
                  *window_size*, *start_time*, *threshold*, and *robust* are
                  ignored for trimming.
                * **ddof** (*int*) -- Degrees of freedom for variance/SEM computation.
                * **stats_method** (*{"non-overlapping", "sliding"}*) -- Block method for statistics.
                * **stats_window_size** (*int or None*) -- Window size for statistics; auto-selected when ``None``.
                * **diagnostics** (*{"compact", "full"}*) -- ``"full"`` includes per-member statistics in Technique 2 output.
                * **verbosity** (*int*) -- ``0`` — silent; ``>0`` — print key steps.
                * **keep_trimmed** (*bool*) -- Include the list of trimmed DataStreams in the result.


   .. py:method:: run(ensemble_or_members)

      Execute the Ensemble Statistics workflow.

      :Parameters: **ensemble_or_members** (*Ensemble or list of DataStream*) -- Raw (untrimmed) ensemble members.

      :returns: *dict* -- Structure depends on *technique*:

                **technique="pooled_block_means"**::

                  {
                    "workflow": "ensemble_statistics",
                    "n_members": int,
                    "column_name": …,
                    "technique": "pooled_block_means",
                    "pooled_block_means": {
                        "trimmed_members": list or None,
                        "statistics": {"results": {col: {…}}, "metadata": {…}},
                        "metadata": {…},
                    },
                    "metadata": {…},
                  }

                **technique="ivw_member_means"** — same shape but
                ``"ivw_member_means"`` key.

                **technique="both"** — both ``"pooled_block_means"`` and
                ``"ivw_member_means"`` keys are present.

                Legacy ``"technique1"`` / ``"technique2"`` inputs are normalised to
                the canonical names; the result dict uses the canonical keys.

      :raises TypeError: If members are not DataStream instances.
      :raises ValueError: If the member list is empty, or all members are empty after
          trimming, or the column is missing.



.. py:class:: RobustWorkflow(operate_safe=True, verbosity=0, drop_fraction=0.25, n_pts_min=100, n_pts_frac_min=0.2, max_lag_frac=0.5, autocorr_sig_level=0.05, decor_multiplier=4.0, std_dev_frac=0.1, fudge_fac=0.1, smoothing_window_correction=0.8, final_smoothing_window=10)

   Set of functions to analyze DataStreams in a robust way.

   This class can handle data streams with a lot of noise and where stationarity or the start
   of steady statistical state (SSS) can be hard to assess. It uses base DataStream methods for statistical
   analysis but adds alternative tools for stationarity assessment and start of SSS detection.

   Note: this class assumes the time points in the data stream are equally spaced in time.

   Core features include:

   - Stationarity assessment that progressively shortens the DataStream to see if the tail
     end of the DataStream is stationary.
   - Start of SSS detection that uses a robust approach based on the smoothed mean of the DataStream.
   - Methods that return "ball park" statistics if the DataStream is not stationary,
     or if there is no SSS segment found.

   :ivar _drop_fraction: DataStream is stationary.
   :vartype _drop_fraction: float, fraction of data to drop from the start of the DataStream to see if the shortened
   :ivar _operate_safe: If True: process data streams in a safe way insisting on stationarity and a segment
                        that is clearly in SSS
                        If False: try to get some results even if the data stream is not stationary or there is no
                        SSS segment found.
   :vartype _operate_safe: bool
   :ivar _verbosity: 0. : very few print statements or plots
                     > 0: more print statements
                     > 1: also show plots of intermediate steps
   :vartype _verbosity: int, level of verbosity for print statements and plots.
   :ivar _drop_fraction: DataStream is stationary.
   :vartype _drop_fraction: float, fraction of data to drop from the start of the DataStream to see if the shortened
   :ivar _n_pts_min:
   :vartype _n_pts_min: int, minimum number of points to keep in the DataStream when shortening it to check for stationarity.
   :ivar _n_pts_frac_min: to check for stationarity.
   :vartype _n_pts_frac_min: float, minimum fraction of the original number of points to keep in the DataStream when shortening it
   :ivar _max_lag_frac: the autocorrelation function to determine the decorrelation length.
   :vartype _max_lag_frac: float, maximum lag (as a fraction of the number of points in the DataStream) to use when computing
   :ivar _autocorr_sig_level: function.
   :vartype _autocorr_sig_level: float, significance level to use when determining the decorrelation length from the autocorrelation
   :ivar _decor_multiplier:
   :vartype _decor_multiplier: float, multiplier to apply to the decorrelation length to get the smoothing window size.
   :ivar _std_dev_frac: of SSS.
   :vartype _std_dev_frac: float, fraction of the std dev of the stationary signal to use as tolerance when determining the start
   :ivar _fudge_fac: used to compute the tolerance for determining the start of SSS.
   :vartype _fudge_fac: float, fudge factor to multiply the initial mean of the smoothed signal with before adding it to the std dev
   :ivar _smoothing_window_correction:
   :vartype _smoothing_window_correction: float, correction factor to apply to the smoothing window size when determining the start of SSS.
   :ivar _final_smoothing_window:
   :vartype _final_smoothing_window: int, smoothing window used to avoid quantities going to zero at end of signal.


   .. py:method:: process_irregular_stream(data_stream, col, start_time = 0.0)

      Process a data stream that is not stationary or has no steady state segment

      :Parameters: * **data_stream** (*DataStream*) -- The data stream to process.
                   * **col** (*str*) -- The column name of the quantity of interest in the data stream.
                   * **start_time** (*float, optional*) -- The time after which to consider data for processing. Default is 0.0.

      :returns: **results_dict** (*dict*) -- Dictionary with results for the quantity of interest.



   .. py:method:: process_data_stream(data_stream_orig, col, start_time = 0.0)

      Process data_stream and handle exceptions gracefully.
      Return mean value and its statistics


      TODO
      * look at number of effective samples we have. Could be low. Allow user to
      override this if they want minimum # of samples for analysis.

      :Parameters: * **data_stream** (*DataStream*) -- The data stream to process.
                   * **col** (*str*) -- The column name of the quantity of interest in the data stream.
                   * **start_time** (*float, optional*) -- The time after which to consider data for processing. Default is 0.0.

      :returns: **results_dict** (*dict*) -- Dictionary with results for the quantity of interest.



   .. py:attribute:: process_data_steam


   .. py:method:: plot_signal_basic_stats(data_stream, col, stats = None, label = None)

      NOTE: make this part of visualization class?

      :Parameters: * **data_stream** (*DataStream*) -- The data stream to plot
                   * **col** (*str*) -- The column name of the quantity to plot in the data stream.
                   * **stats** (*dict, optional*) -- Dictionary with statistics returned by process_data_stream(). Default is None.
                   * **label** (*str, optional*) -- Label to use in title of graph. Default is None.

      :returns: *shows a plot of the signal with mean, confidence interval and start of SSS (if stats provided)*



