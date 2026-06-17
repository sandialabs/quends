quends
======

.. py:module:: quends


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/quends/base/index
   /autoapi/quends/cli/index
   /autoapi/quends/postprocessing/index
   /autoapi/quends/preprocessing/index
   /autoapi/quends/workflow/index


Attributes
----------

.. autoapisummary::

   quends.RollingVarianceTrimStrategy
   quends.SSSStartTrimStrategy
   quends.StandardDeviationTrimStrategy
   quends.ThresholdTrimStrategy


Classes
-------

.. autoapisummary::

   quends.DataStream
   quends.Ensemble
   quends.DataStreamOperation
   quends.MakeDataStreamStationaryOperation
   quends.IQRTrimStrategy
   quends.MeanVariationTrimStrategy
   quends.NoiseThresholdTrimStrategy
   quends.QuantileTrimStrategy
   quends.RollingVarianceThresholdTrimStrategy
   quends.SelfConsistentTrimStrategy
   quends.TrimDataStreamOperation
   quends.TrimStrategy
   quends.Exporter
   quends.Plotter
   quends.BatchEnsembleWorkflow
   quends.EnsembleAverageWorkflow
   quends.EnsembleStatisticsWorkflow
   quends.RobustWorkflow


Functions
---------

.. autoapisummary::

   quends.build_trim_strategy
   quends.from_csv
   quends.from_dict
   quends.from_gx
   quends.from_json
   quends.from_netcdf
   quends.from_numpy


Package Contents
----------------

.. py:class:: DataStream(data, history = None)

   .. py:property:: data
      :type: Any


      The underlying pandas DataFrame.


   .. py:property:: df
      :type: Any


      Backward-compatible alias for the underlying pandas DataFrame.


   .. py:property:: history
      :type: quends.base.history.DataStreamHistory



   .. py:method:: head(n = 5)


   .. py:method:: variables()

      Return all column names in the underlying DataFrame (including 'time').

      To obtain only signal columns use::

          [c for c in ds.variables() if c != "time"]

      :returns: *pandas.Index* -- All column names in ``self.data``.



   .. py:method:: mean(column_name=None, method='non-overlapping', window_size=None)

      Compute block or sliding window means for each column.

      Thin wrapper over :meth:`compute_statistics` — extracts ``mean`` and
      ``window_size`` so callers that only need the mean don't have to
      unpack the full statistics dict.



   .. py:method:: mean_uncertainty(column_name=None, ddof=1, method='non-overlapping', window_size=None)

      Estimate the standard error of the mean via Geyer ESS on block means.

      Thin wrapper over :meth:`compute_statistics` — extracts
      ``mean_uncertainty`` and ``window_size``.



   .. py:method:: confidence_interval(column_name=None, ddof=1, method='non-overlapping', window_size=None, confidence_level = 0.95, ci_method = 'normal')

      Build confidence intervals around block/sliding means.

      Thin wrapper over :meth:`compute_statistics` — extracts
      ``confidence_interval`` and ``window_size``.  Columns with no valid
      data propagate the error dict rather than raising ``KeyError``.

      See :meth:`compute_statistics` for the meaning of *confidence_level*
      and *ci_method*.  Defaults preserve the historical 95 % normal CI
      (multiplier ``1.96``).



   .. py:method:: trim(column_name=None, *, method = 'std', window_size = 10, start_time = 0.0, threshold = None, robust = True, **strategy_kwargs)

      Trim this stream to its steady state and return a new ``DataStream``.

      Convenience one-liner over :func:`quends.base.trim.build_trim_strategy` +
      :class:`~quends.base.trim.TrimDataStreamOperation` (the explicit/low-level
      path still works exactly as before — this just wraps it).

      :Parameters: * **column_name** (*str, optional*) -- Column to detect steady state on. If ``None`` and the stream has a
                     single non-``time`` column, that column is used automatically.
                   * **method** (*str*) -- ``"std"`` | ``"threshold"`` | ``"rolling_variance"`` |
                     ``"self_consistent"`` | ``"iqr"`` | ``"mean_variation"``.
                   * **window_size, start_time, threshold, robust** -- Strategy parameters (see ``build_trim_strategy``).
                   * **\*\*strategy_kwargs** -- Extra attributes set on the strategy (e.g. ``drop_leading_nonpositive=False``).

      :returns: *DataStream* -- The trimmed stream (empty if no steady state was detected).



   .. py:method:: compute_statistics(column_name=None, ddof=1, method='non-overlapping', window_size=None, confidence_level = 0.95, ci_method = 'normal')

      Aggregate statistics for each column using autotuned independent block means.

      Window selection and block-mean computation go through :meth:`_process_column`
      → :func:`~quends.base.utils.autotune_blocks` (the single canonical helper
      shared with the ensemble pipeline).

      :Parameters: * **column_name** (*str or list or None*)
                   * **ddof** (*int*)
                   * **method** (*{'non-overlapping', 'sliding'}*) -- Block type.  Independence autotuning always uses non-overlapping blocks
                     regardless of this setting.
                   * **window_size** (*int or None*) -- User-supplied window; triggers autotune when ``None``.
                   * **confidence_level** (*float*) -- Two-sided confidence level for the CI.  Default ``0.95``.
                   * **ci_method** (*{'normal', 't'}*) -- CI quantile family.  Default ``'normal'`` (preserves the historical
                     ``1.96`` multiplier exactly for backward compatibility).  When ``'t'``,
                     uses Student's *t* with ``dof = max(1, se_effective_n - 1)``.

      :returns: *dict* -- ``{col: {…statistics…}}`` with the following canonical keys per column:

                ``mean``, ``mean_uncertainty`` (SEM), ``variance``, ``confidence_interval``,
                ``pm_std``, ``effective_sample_size`` (Geyer ESS on raw series),
                ``window_size``, ``n_short_averages`` (number of block means),
                ``ess_blocks`` (Geyer ESS on block means), ``se_effective_n``,
                ``se_method``, ``independence_status``, ``independent``,
                ``ljungbox_lags`` (list), ``ljungbox_pvalues`` (list),
                ``ljungbox_pvalue`` (scalar min — convenience alias matching ensemble output),
                ``ci_method``, ``confidence_level``,
                ``warning`` (if applicable).

                On error: ``{col: {"error": "…"}}``



   .. py:method:: cumulative_statistics(column_name=None, method='non-overlapping', window_size=None)

      Generate cumulative mean and uncertainty time series for each column.

      Returns per-column cumulative arrays plus ``window_size``.

      .. rubric:: Notes

      ``cumulative_uncertainty`` is the expanding **standard deviation** of the
      processed series, while ``standard_error`` is the expanding SEM
      (std / sqrt(count)). Use ``standard_error`` for uncertainty-on-the-mean.



   .. py:method:: additional_data(column_name=None, ddof=1, method='sliding', window_size=None, reduction_factor=0.1)

      Estimate additional sample size needed to reduce SEM by `reduction_factor` via power-law fit.

      Returns model parameters and sample projections.

      .. rubric:: Notes

      The power law is currently fit to ``cumulative_statistics``'
      ``cumulative_uncertainty`` series. See ``cumulative_statistics`` — that
      key holds the expanding standard deviation, not the SEM; fitting a
      shrinking-SEM power law to it is a known limitation (see AUDIT_REPORT H2).



   .. py:method:: is_stationary(columns)

      Perform Augmented Dickey-Fuller test for each specified column.

      :Parameters: **columns** (*str or list of str*)

      :returns: *dict* -- {column: True if stationary (p < 0.05), else False}



   .. py:method:: get_block_effective_n(column_name, method = 'non-overlapping', window_size = None)

      Return Geyer ESS on block means for one column.

      Thin wrapper over :meth:`compute_statistics` — extracts the
      ``ess_blocks``, ``window_size``, and ``n_short_averages`` fields so
      that callers that only need block-level ESS info don't have to unpack
      the full statistics dict.

      :returns: *dict* -- ``{"effective_n": float, "window_size": int, "n_blocks": int}``



   .. py:method:: effective_sample_size(column_names=None, alpha=0.05)

      Compute ESS using Geyer positive-pair truncation of the ACF.

      The integrated autocorrelation time ``tau_int`` is estimated by summing
      consecutive positive pairs of the normalised ACF and truncating as soon
      as a pair turns non-positive.  ESS is then ``n / tau_int``.

      :Parameters: * **column_names** (*str or list of str or None*) -- Columns to compute ESS for; defaults to all non-time columns.
                   * **alpha** (*float*) -- Reserved for API compatibility; not used in the Geyer estimator.

      :returns: *dict* -- ``{'results': {col: ESS_int or message_dict}}``



   .. py:method:: robust_effective_sample_size(x, rank_normalize=True, min_samples=8, return_relative=False)
      :staticmethod:


      Compute a robust ESS via pairwise autocorrelations and optional rank-normalization.

      :Parameters: * **x** (*array-like*)
                   * **rank_normalize** (*bool*)
                   * **min_samples** (*int*)
                   * **return_relative** (*bool*)

      :returns: *float or tuple* -- ESS (and ESS/n ratio if return_relative).



   .. py:method:: ess_robust(column_names=None, rank_normalize=False, min_samples=8, return_relative=False)

      Wrapper for `robust_effective_sample_size` over multiple columns.

      :Parameters: * **column_names** (*str or list or None*)
                   * **rank_normalize** (*bool*)
                   * **min_samples** (*int*)
                   * **return_relative** (*bool*)

      :returns: *dict* -- {'results': {col: ESS or tuple}}



   .. py:method:: normalize_data(df)
      :staticmethod:


      Min-Max normalize all signal columns (excluding 'time') to [0,1].

      Operates on a copy; the input DataFrame is not mutated.

      :Parameters: **df** (*pandas.DataFrame*)

      :returns: *pandas.DataFrame*



.. py:class:: Ensemble(data_streams)

   Manages an ensemble of DataStream instances for multi-stream analysis.

   Core logic is implemented in :mod:`quends.base.ensemble_utils` and
   :mod:`quends.base.ensemble_statistics`.  The methods here are thin wrappers
   that delegate to those module-level functions so that workflow classes can
   call the same logic without going through this class.


   .. py:attribute:: data_streams


   .. py:method:: from_files(paths, variable, *, loader=None)
      :classmethod:


      Build an Ensemble by loading one variable from each file.

      Convenience constructor that replaces the common
      ``[from_csv(p, var) for p in paths]`` boilerplate. ``.nc`` files are
      loaded with :func:`quends.preprocessing.from_netcdf`, everything else
      with :func:`quends.preprocessing.from_csv`, unless an explicit ``loader``
      callable ``loader(path, variable) -> DataStream`` is given.

      :Parameters: * **paths** (*iterable of str or pathlib.Path*) -- File paths to load (one ensemble member each).
                   * **variable** (*str*) -- The variable to load from every file.
                   * **loader** (*callable, optional*) -- Override the auto-selected loader.

      :returns: *Ensemble*



   .. py:method:: head(n = 5)


   .. py:method:: get_member(index)


   .. py:method:: members()


   .. py:method:: common_variables()

      Column names shared by all members, excluding 'time'.



   .. py:method:: summary(verbose = False)

      Return a summary dict of the ensemble.

      :Parameters: **verbose** (*bool*) -- If True, also print a short header. Default False — this is a pure
                   getter and should not print as a side effect.



   .. py:method:: collect_histories(ds_list)
      :staticmethod:



   .. py:method:: check_time_steps_uniformity(tol = 1e-08, verbose = False)

      Inspect the time-step regularity of each ensemble member.

      For each member, compute diffs of the 'time' column and classify as:
        - "AllEqual"         all steps identical (within tol)
        - "AllEqualButLast"  all steps equal except the last one
        - "NotUniform"       multiple distinct step sizes

      :returns: *dict* --

                Mapping with the structure::

                    {
                      "uniform": bool,          # all members AllEqual with the same step
                      "majority_step": float,   # most common step across all members
                      "members": {
                        "Member i": {
                          "status": str,
                          "unique_steps": list[float],
                          "n_steps": int,
                          "t_min": float,
                          "t_max": float,
                        }
                      }
                    }

      .. rubric:: Notes

      Delegates to :func:`~quends.base.ensemble_utils.check_time_steps_uniformity`.



   .. py:method:: interpolate_to_common_time(method = 'spline', tol = 1e-08, verbose = False)

      Interpolate all ensemble members onto a common, regular time grid.

      The common grid spans [min(t_start), max(t_end)] across all members
      using the majority time step.

      :Parameters: * **method** (*{"spline", "linear"}*) -- Interpolation method: cubic spline or linear.
                   * **tol** (*float*) -- Tolerance for step-size uniformity check.
                   * **verbose** (*bool*) -- Print grid diagnostics.

      :returns: *(interpolated_ensemble, diagnostics_dict)*

      .. rubric:: Notes

      Delegates to
      :func:`~quends.base.ensemble_utils.interpolate_to_common_time`.



   .. py:method:: compute_average_ensemble(members = None, interp_method = 'spline', tol = 1e-08, min_coverage = 1, verbose = False)

      Build a single averaged DataStream from ensemble members.

      If all members share the same time grid (detected via
      check_time_steps_uniformity), averages directly.
      If grids differ, interpolates all members to a common grid first.

      :Parameters: * **members** (*list of DataStream, optional*) -- Subset to average; defaults to all.
                   * **interp_method** (*{"spline", "linear"}*) -- Interpolation method when grids differ.
                   * **tol** (*float*) -- Tolerance for uniformity check.
                   * **min_coverage** (*int*) -- Minimum number of members that must contribute to a time point.
                   * **verbose** (*bool*) -- Print diagnostics.

      :returns: *DataStream* -- Single averaged trace.

      .. rubric:: Notes

      Delegates to
      :func:`~quends.base.ensemble_utils.compute_average_ensemble`.



   .. py:method:: trim(column_name, method = 'std', window_size = 10, start_time = 0.0, threshold = None, robust = True, **kwargs)

      Thin wrapper: trim each ensemble member using the unified trim strategy system.

      Uses :func:`~quends.base.trim.build_trim_strategy` and
      :class:`~quends.base.trim.TrimDataStreamOperation` from ``trim.py``
      directly — the same canonical low-level path used for all trimming.
      No separate trimming logic is defined here.

      :Parameters: * **column_name** (*str*) -- Column whose steady-state start drives the trim.
                   * **method** (*str*) -- Trim strategy name: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                     ``"self_consistent"``, or ``"iqr"``.
                   * **window_size** (*int*) -- Block / rolling-window size (passed through to the strategy).
                     Formerly named ``batch_size`` — that name is still accepted as a
                     deprecated keyword argument for backward compatibility.
                   * **start_time** (*float*) -- Ignore data before this simulation time.
                   * **threshold** (*float or None*)
                   * **robust** (*bool*)

      :returns: *Ensemble* -- A new Ensemble containing only the members that returned a
                non-empty trimmed DataStream.

      :raises ValueError: If *method* is unrecognised, or if every member produced an empty
          result (no steady state found in any member).

      .. rubric:: Notes

      Backward compatibility: ``batch_size`` is silently mapped to
      ``window_size`` so that existing callers are not broken.

      .. rubric:: Examples

      >>> trimmed_ens = ens.trim("HeatFlux_st", method="std", window_size=20)
      >>> trimmed_ens = ens.trim("HeatFlux_st", method="iqr", threshold=0.05)



   .. py:method:: is_stationary(columns)

      Test stationarity for columns across all members.

      Returns an enriched per-member report (unlike DataStream.is_stationary
      which returns a simple {col: bool} dict).

      :returns: *dict* --

                {"results": {"Member i": {col: bool, ...}, ...},
                 "metadata": {"Member i": {}, ...}}



   .. py:method:: mean(column_name=None, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact')

      Ensemble mean.

      technique
          ``"ensemble_average"``     — build single averaged trace, compute stats on it
          ``"pooled_block_means"``   — preferred for trimmed ensembles
          ``"ivw_member_means"``     — member-wise then inverse-variance aggregate
          (legacy ``0/1/2`` and ``"technique0"/1/2`` strings still accepted)



   .. py:method:: mean_uncertainty(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact')

      Ensemble SEM. See `mean()` for technique semantics.



   .. py:method:: confidence_interval(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact')

      Ensemble confidence interval. See `mean()` for technique semantics.



   .. py:method:: compute_statistics(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal')

      Aggregate mean, SEM, CI, ±SEM, variance, ESS across the ensemble.

      technique=``"ensemble_average"``   Single averaged trace, then DataStream stats.
      technique=``"pooled_block_means"`` Preferred for trimmed ensembles.
      technique=``"ivw_member_means"``   Per-member stats, inverse-variance combined.
      (Legacy ``0``/``1``/``2`` and ``"technique0"``/``"technique1"``/``"technique2"``
      also accepted.)

      Confidence-interval parameters (defaults preserve historical 95 %
      normal CI, multiplier ``1.96``):

      confidence_level : float
          Two-sided confidence level.
      ci_method : {'normal', 't'}
          Quantile family.  ``'t'`` is supported for techniques 0 and 1
          (where dof is well-defined); raises for technique 2.

      Returns {"results": {col: {stats}}, "metadata": {...}}.

      .. rubric:: Notes

      Delegates to
      :func:`~quends.base.ensemble_statistics.compute_ensemble_statistics`.



   .. py:method:: compute_uncertainty(method='pooled_block_means', column_name=None, *, ddof = 1, window_size = None, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal')

      Friendly alias for :meth:`compute_statistics` keyed by estimator name.

      ``method`` is the estimator: ``"ensemble_average"`` | ``"pooled_block_means"``
      | ``"ivw"`` (plus the legacy ``technique`` aliases). Equivalent to calling
      ``compute_statistics(..., technique=method)`` — the latter still works.

      Returns the same ``{"results": {...}, "metadata": {...}}`` schema.



   .. py:method:: effective_sample_size(column_names=None, alpha = 0.05, technique=POOLED_BLOCK_MEANS)

      Compute ESS via ensemble statistics (delegates to compute_statistics).



   .. py:method:: effective_sample_size_blocks(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS)

      ESS on block means (Geyer) from compute_statistics.



   .. py:method:: n_short_averages(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS)

      Count of block means from compute_statistics.



.. py:class:: DataStreamOperation(operation_name = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:property:: name
      :type: str



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



.. py:class:: IQRTrimStrategy(window_size = 10, start_time = 0.0, threshold = 0.05)

   Bases: :py:obj:`TrimStrategy`


   Trim using IQR-based steady-state detection:
   IQR(remaining) <= threshold * abs(median(remaining)).

   :Parameters: * **window_size** (*int*) -- Minimum samples before evaluating (used for loop start only).
                * **threshold** (*float*) -- Fraction of abs(median) that IQR must fall below.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


.. py:class:: MeanVariationTrimStrategy(*, max_lag_frac=None, verbosity=None, autocorr_sig_level=None, decor_multiplier=None, std_dev_frac=None, fudge_fac=None, smoothing_window_correction=None, final_smoothing_window=None)

   Bases: :py:obj:`TrimStrategy`


   Trim using Statistical Steady State detection.


   .. py:attribute:: max_lag_frac
      :value: None



   .. py:attribute:: verbosity
      :value: None



   .. py:attribute:: autocorr_sig_level
      :value: None



   .. py:attribute:: decor_multiplier
      :value: None



   .. py:attribute:: std_dev_frac
      :value: None



   .. py:attribute:: fudge_fac
      :value: None



   .. py:attribute:: smoothing_window_correction
      :value: None



   .. py:attribute:: final_smoothing_window
      :value: None



   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


   .. py:method:: apply(data_stream, column_name, **kwargs)

      Identify and trim the signal to the start of the Statistical Steady State (SSS)

      :Parameters: * **col** (*str*) -- The name of the column in `data_stream.data` to analyze for steady state.
                   * **workflow** (*object*) -- A configuration/workflow object containing parameters:
                     - `_max_lag_frac`: Fraction of data used for autocorrelation lag.
                     - `_verbosity`: Integer controlling plot and print output levels.
                     - `_autocorr_sig_level`: Significance level for the Z-test on lags.
                     - `_decor_multiplier`: Multiplier for the calculated decorrelation length.
                     - `_std_dev_frac`: Fraction of standard deviation used for tolerance.
                     - `_fudge_fac`: Constant to prevent zero-tolerance in noiseless signals.
                     - `_smoothing_window_correction`: Factor to adjust for rolling mean lag.
                     - `_final_smoothing_window`: Window size for smoothing the metric curves.

      :returns: *DataStream* -- A new DataStream object containing the DataFrame trimmed to the SSS start.
                Returns an empty DataFrame if no SSS is identified.



.. py:class:: NoiseThresholdTrimStrategy(window_size = 10, start_time = 0.0, threshold = None, robust = True)

   Bases: :py:obj:`TrimStrategy`


   Trim using rolling standard deviation on normalized data.


   .. py:method:: apply(data_stream, column_name, **kwargs)

      Template method that defines the trimming workflow.

      ``_stationary_checked`` lets a subclass that has already verified
      stationarity (e.g. :class:`NoiseThresholdTrimStrategy`) skip the
      redundant re-check here without changing behavior.



   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


.. py:class:: QuantileTrimStrategy(window_size = 10, start_time = 0.0, robust = True)

   Bases: :py:obj:`TrimStrategy`


   Trim based on standard-deviation / robust MAD steady-state criteria.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


.. py:class:: RollingVarianceThresholdTrimStrategy(window_size = 50, start_time = 0.0, robust = True, threshold = 0.1)

   Bases: :py:obj:`TrimStrategy`


   Detect steady-state when the rolling spread falls below a threshold.

   Note: despite the historical name ("variance"), the criterion uses the
   rolling **standard deviation** (``.std()``) compared against
   ``threshold * mean(rolling_std)``. The name is retained for backward
   compatibility (``build_trim_strategy("rolling_variance")``).


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


.. py:data:: RollingVarianceTrimStrategy

.. py:data:: SSSStartTrimStrategy

.. py:class:: SelfConsistentTrimStrategy(window_size = 10, start_time = 0.0, robust = True, rel_tol_mu = 0.1, rel_tol_sigma = 0.05)

   Bases: :py:obj:`TrimStrategy`


   Trim using self-consistent block comparison:
   find the earliest time where consecutive non-overlapping blocks of size W
   agree in both mean and spread (within relative tolerances).

   :Parameters: * **window_size** (*int*) -- Block size W.
                * **robust** (*bool*) -- Use median + MAD (scaled) vs mean + std.
                * **rel_tol_mu** (*float*) -- Relative tolerance for mean comparison across blocks.
                * **rel_tol_sigma** (*float*) -- Relative tolerance for spread comparison across blocks.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


.. py:data:: StandardDeviationTrimStrategy

.. py:data:: ThresholdTrimStrategy

.. py:class:: TrimDataStreamOperation(strategy, operation_name = 'trim')

   Bases: :py:obj:`quends.base.operations.DataStreamOperation`


   Operation that applies a TrimStrategy to a DataStream.


   .. py:property:: strategy
      :type: TrimStrategy



.. py:class:: TrimStrategy(window_size = 10, start_time = 0.0, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Abstract base class describing a trim strategy.


   .. py:attribute:: window_size
      :value: 10



   .. py:attribute:: start_time
      :value: 0.0



   .. py:property:: method_name
      :type: str

      :abstractmethod:


      Return the method name for this strategy.


   .. py:method:: apply(data_stream, column_name, _stationary_checked = False, **kwargs)

      Template method that defines the trimming workflow.

      ``_stationary_checked`` lets a subclass that has already verified
      stationarity (e.g. :class:`NoiseThresholdTrimStrategy`) skip the
      redundant re-check here without changing behavior.



.. py:function:: build_trim_strategy(method = 'std', window_size = 10, start_time = 0.0, threshold = None, robust = True)

   Factory: map a method string to a configured :class:`TrimStrategy` instance.

   This is the **single canonical source of truth** for the method-string →
   strategy-class mapping.  All convenience wrappers (``DataStream.trim``,
   ``Ensemble.trim``, ``Plotter._trim_datastream``) delegate here so the
   mapping never drifts apart.

   :Parameters: * **method** (*str*) -- One of ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                  ``"self_consistent"``, ``"iqr"``.
                * **window_size** (*int*) -- Block / window size passed to the strategy.
                * **start_time** (*float*) -- Ignore data before this time.
                * **threshold** (*float or None*) -- Required for ``"threshold"``; optional for ``"rolling_variance"``
                  (default 0.1) and ``"iqr"`` (default 0.05).
                * **robust** (*bool*) -- Use median/MAD instead of mean/std where applicable.

   :returns: *TrimStrategy*

   :raises ValueError: If *method* is not one of the recognised strings.

   .. rubric:: Examples

   >>> strategy = build_trim_strategy("std", window_size=20)
   >>> op = TrimDataStreamOperation(strategy)
   >>> trimmed = op(ds, column_name="Q")


.. py:class:: Exporter(output_dir='exported_results', overwrite=False)

   Export data/results in various formats: DataFrame/CSV, JSON, dictionary, and
   NumPy array. Provides both display (to console) and save (to file) helpers,
   with automatic conversion of NumPy types to native Python types.

   Safety
   ------
   By default the Exporter does **not** overwrite existing files: ``save_*`` /
   ``export_*`` raise :class:`FileExistsError` if the target already exists.
   Pass ``overwrite=True`` (per call) or construct with ``overwrite=True`` to
   allow clobbering. This protects previously-saved results from silent loss
   when a study is re-run.


   .. py:attribute:: output_dir
      :value: 'exported_results'



   .. py:attribute:: overwrite
      :value: False



   .. py:method:: to_native_types(obj)
      :staticmethod:


      Recursively convert NumPy scalar types in dicts/lists/tuples to native
      Python types. Thin wrapper over :func:`quends.base.utils.to_native_types`.



   .. py:method:: to_dataframe(data)

      Convert input data to a pandas DataFrame.



   .. py:method:: to_dictionary(data)

      Convert input data to a dictionary with native Python types.



   .. py:method:: to_numpy(data)

      Convert input data to a NumPy array.



   .. py:method:: to_json(data)

      Convert input data to a JSON string with native Python types.



   .. py:method:: display_dataframe(data, head=None)

      Display data as a DataFrame.



   .. py:method:: display_dictionary(data)

      Display data as a (native-typed) dictionary.



   .. py:method:: display_numpy(data)

      Display data as a NumPy array.



   .. py:method:: display_json(data)

      Display data as a (native-typed) JSON string.



   .. py:method:: save_dataframe(data, file_name='dataframe.csv', *, overwrite=None)

      Save data as a CSV file. Returns the written path.



   .. py:method:: save_dictionary(data, file_name='data_dictionary.json', *, overwrite=None)

      Save data as a JSON dictionary file. Returns the written path.



   .. py:method:: save_numpy(data, file_name='data.npy', *, overwrite=None)

      Save data as a ``.npy`` file. Returns the written path.



   .. py:method:: save_json(data, file_name='data.json', *, overwrite=None)

      Save data as a JSON file. Returns the written path.



   .. py:method:: export_figure(fig, filename='figure.png', dpi=300, *, overwrite=None)

      Save a Matplotlib figure. Returns the written path.



   .. py:method:: save_results(results, name, *, overwrite=None, metadata=None)

      Save a results object as CSV (if tabular) or JSON, plus a provenance
      sidecar ``<name>.meta.json``.

      :Parameters: * **results** (*DataFrame | dict | ndarray*) -- The results to persist.
                   * **name** (*str*) -- Base name (without extension). A data file and a ``<name>.meta.json``
                     sidecar are written.
                   * **overwrite** (*bool, optional*) -- Per-call overwrite policy (defaults to the instance policy).
                   * **metadata** (*dict, optional*) -- Provenance to record in the sidecar (source file, variable, trim
                     parameters, etc.). ``schema_version`` is added automatically.

      :returns: *(data_path, meta_path)*



   .. py:method:: export_dataframe(data, filename='dataframe.csv', *, overwrite=None)

      Deprecated alias for :meth:`save_dataframe` (kept for compatibility).



.. py:class:: Plotter(output_dir='results_figures')

   Plotting utilities for DataStream and Ensemble objects.

   Uniform plotting contract
   --------------------------
   Every public plotting method accepts the same output-control keywords and
   returns the Matplotlib objects instead of forcing display:

   * ``save`` (bool)        — write the figure to disk.
   * ``show`` (bool)        — call ``plt.show()`` (default **False**; scripts/CI safe).
   * ``filename`` (str)     — output filename (single-figure methods).
   * ``output_dir`` (str)   — per-call output directory (defaults to the ctor dir).
   * ``overwrite`` (bool)   — allow clobbering an existing file (default False).
   * ``dpi`` (int)          — raster resolution when saving.

   Return value:

   * single-figure methods return ``(fig, axes)``;
   * multi-figure methods (one figure per variable/member) return a list of
     ``(fig, axes)`` tuples.

   Regenerate-from-CSV
   -------------------
   The stats/trim-heavy methods accept optional *precomputed* arguments
   (``stats=``, ``ss_starts=``, ``means=``, ``avg_df=``, ``acf_values=``). When
   supplied, the method skips the expensive computation and only renders — so a
   figure can be rebuilt from saved results without recomputing statistics.


   .. py:attribute:: output_dir
      :value: 'results_figures'



   .. py:method:: format_dataset_name(dataset_name)
      :staticmethod:


      Return a title-cased, space-separated version of a dataset name.



   .. py:method:: trace_plot(data, variables_to_plot=None, *, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Plot raw time-series traces. Returns a list of (fig, axes).



   .. py:method:: trace_plot_with_mean(data, variables_to_plot=None, window_size=None, *, stats=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Plot each trace with block-mean + 95% CI overlaid.

      Pass ``stats={column: {"mean": .., "confidence_interval": (lo, hi)}}`` to
      skip ``compute_statistics`` (regenerate-from-saved). Returns list of (fig, axes).



   .. py:method:: ensemble_trace_plot(data, variables_to_plot=None, *, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Overlay traces from all members, one figure per variable. Returns list of (fig, axes).



   .. py:method:: ensemble_trace_plot_with_mean(data, variables_to_plot=None, window_size=None, *, means=None, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Overlay member traces with per-member block mean.

      Pass ``means={dataset_name: {var: mean}}`` to skip ``compute_statistics``.
      Returns list of (fig, axes).



   .. py:method:: steady_state_automatic_plot(data, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, *, ss_starts=None, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Auto-detect steady-state start per variable and annotate.

      Pass ``ss_starts={column: ss_start_time}`` to skip trimming
      (regenerate-from-saved). Returns list of (fig, axes).



   .. py:method:: steady_state_plot(data, variables_to_plot=None, steady_state_start=None, *, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Annotate steady state from a user-supplied start (float or {var: float}).

      Returns list of (fig, axes).



   .. py:method:: plot_acf(data, alpha=0.05, column=None, ax=None, *, acf_values=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Plot the autocorrelation function.

      Pass ``acf_values`` (precomputed array) to skip ``acf()``. If ``ax`` is
      given the stem is drawn into it (no save/show); otherwise a new figure is
      created and returned as (fig, ax).



   .. py:method:: plot_acf_ensemble(ensemble_obj, alpha=0.05, column=None, *, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      ACF grid, one subplot per member. Returns (fig, axes).



   .. py:method:: ensemble_steady_state_automatic_plot(ensemble_obj, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, *, ss_starts=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Auto-detect steady state per member, one subplot each. Returns (fig, axes).

      Pass ``ss_starts={member_index: {var: ss_start}}`` to skip trimming.



   .. py:method:: ensemble_steady_state_plot(ensemble_obj, variables_to_plot=None, steady_state_start=None, *, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Annotate each member with a user-supplied SS start. Returns (fig, axes).



   .. py:method:: plot_ensemble(ensemble_obj, variables_to_plot=None, *, avg_df=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Members + ensemble average, 2-column grid. Returns (fig, axes).

      Pass ``avg_df`` (a precomputed average DataFrame) to skip
      ``compute_average_ensemble``.



   .. py:method:: plot_ensemble_with_average(ensemble_obj, variables_to_plot=None, *, avg_df=None, condensed_legend=False, y_range=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Members + ensemble average with optional condensed legend / y-range.

      Pass ``avg_df`` to skip ``compute_average_ensemble``. Returns (fig, axes).



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


.. py:function:: from_gx(file, variable)

   Load a single variable from a GX output file (.nc or .csv).

   :Parameters: * **file** (*str*) -- Path to the GX output file (``.nc`` or ``.csv``).
                * **variable** (*str*) -- The variable name to load (e.g. ``"HeatFlux_st"``).

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If no variable is specified or the file format is unsupported.


.. py:function:: from_json(file, variable)

   Load a single variable as a data stream from a JSON file.

   The JSON may be either an array of records or an object with a top-level
   ``"data"`` key holding such an array. The returned :class:`DataStream`
   contains the ``time`` column (when present) together with the requested
   ``variable`` column.

   :Parameters: * **file** (*str*) -- The path to the JSON file.
                * **variable** (*str*) -- The column name to load. Must exist in the JSON file.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]`` (or just
             ``[variable]`` if no ``time`` column is present).

   :raises ValueError: If the file does not exist or the variable is not found.


.. py:function:: from_netcdf(file, variable)

   Load a single variable from a NetCDF4 file into a data stream.

   Variables that end with ``'_t'`` or ``'_st'`` are extracted from the
   ``Diagnostics`` group and aligned to the ``time`` grid. The returned
   :class:`DataStream` contains the ``time`` column together with the requested
   ``variable`` column.

   :Parameters: * **file** (*str*) -- Path to the NetCDF4 file.
                * **variable** (*str*) -- The variable name to load. Must exist in the file.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If the file does not exist or the variable is not found.


.. py:function:: from_numpy(np_array, variable, time=None)

   Load a single-variable data stream (with a time column) from a NumPy array.

   To match the other loaders, the returned :class:`DataStream` carries a
   ``time`` column alongside the named ``variable``. Because a NumPy array has
   no intrinsic time axis, the time values come from one of three sources:

   * **1D array + ``time``** — the supplied ``time`` array (same length).
   * **1D array, no ``time``** — a synthesized integer index ``0, 1, 2, ...``.
   * **Nx2 array** — interpreted as two columns; the (strictly increasing)
     column is taken as ``time`` and the other as ``variable``. If neither is
     monotonic, the first column is assumed to be time.

   :Parameters: * **np_array** (*np.ndarray*) -- A 1D array of values, or an Nx2 array
                  ``[time, variable]``.
                * **variable** (*str*) -- The column name to assign to the data values.
                * **time** (*array-like, optional*) -- Explicit time values for the 1D case
                  (must match the length of ``np_array``). Ignored for Nx2 input.

   :returns: *DataStream* -- A DataStream containing ``[time, variable]``.

   :raises ValueError: If the input is not a NumPy array of a supported shape, or
       if a supplied ``time`` array has the wrong length.


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



