quends.base
===========

.. py:module:: quends.base


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/quends/base/data_stream/index
   /autoapi/quends/base/ensemble/index
   /autoapi/quends/base/ensemble_statistics/index
   /autoapi/quends/base/ensemble_utils/index
   /autoapi/quends/base/history/index
   /autoapi/quends/base/operations/index
   /autoapi/quends/base/stationary/index
   /autoapi/quends/base/trim/index
   /autoapi/quends/base/utils/index


Attributes
----------

.. autoapisummary::

   quends.base.RollingVarianceTrimStrategy
   quends.base.SSSStartTrimStrategy
   quends.base.StandardDeviationTrimStrategy
   quends.base.ThresholdTrimStrategy


Classes
-------

.. autoapisummary::

   quends.base.DataStream
   quends.base.Ensemble
   quends.base.DataStreamHistory
   quends.base.DataStreamHistoryEntry
   quends.base.DataStreamOperation
   quends.base.MakeDataStreamStationaryOperation
   quends.base.IQRTrimStrategy
   quends.base.MeanVariationTrimStrategy
   quends.base.NoiseThresholdTrimStrategy
   quends.base.QuantileTrimStrategy
   quends.base.RollingVarianceThresholdTrimStrategy
   quends.base.SelfConsistentTrimStrategy
   quends.base.TrimDataStreamOperation
   quends.base.TrimStrategy


Functions
---------

.. autoapisummary::

   quends.base.build_trim_strategy


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
         :class: example

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



.. py:class:: DataStreamHistory(entries = None)

   Ordered collection of operations applied to a DataStream.


   .. py:method:: append(entry)

      Append ``entry`` to this history and return self for chaining.



   .. py:method:: copy()

      Return an independent history object with the same immutable entries.



   .. py:method:: entries()

      Expose the ordered sequence of history entries.



.. py:class:: DataStreamHistoryEntry

   Immutable record describing a single operation performed on a data stream.


   .. py:attribute:: operation_name
      :type:  str


   .. py:attribute:: parameters
      :type:  Mapping[str, Any]


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
      :class: example

   >>> strategy = build_trim_strategy("std", window_size=20)
   >>> op = TrimDataStreamOperation(strategy)
   >>> trimmed = op(ds, column_name="Q")


