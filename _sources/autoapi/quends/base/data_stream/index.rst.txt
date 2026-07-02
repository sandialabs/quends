quends.base.data_stream
=======================

.. py:module:: quends.base.data_stream


Classes
-------

.. autoapisummary::

   quends.base.data_stream.DataStream


Module Contents
---------------

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



