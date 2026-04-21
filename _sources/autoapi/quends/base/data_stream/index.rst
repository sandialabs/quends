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



   .. py:property:: history
      :type: quends.base.history.DataStreamHistory



   .. py:method:: head(n = 5)


   .. py:method:: variables()

      List the signal variable (column) names, excluding the 'time' column.

      Returns
      -------
      Index
          ColumnIndex of variable names in `self.df`.



   .. py:method:: mean(column_name=None, method='non-overlapping', window_size=None)

      Compute block or sliding window means for each column.

      Private helper for compute_statistics and confidence intervals.



   .. py:method:: mean_uncertainty(column_name=None, ddof=1, method='non-overlapping', window_size=None)

      Estimate the standard error of the mean via block/sliding windows.

      Private helper.



   .. py:method:: confidence_interval(column_name=None, ddof=1, method='non-overlapping', window_size=None)

      Build 95% confidence intervals around block/sliding means.

      Private helper.



   .. py:method:: compute_statistics(column_name=None, ddof=1, method='non-overlapping', window_size=None)

      Aggregate statistics: mean, uncertainty, CI, pm_std bounds, ESS, and window size.

      Appends the operation to history and embeds deduplicated metadata in the results.

      Parameters
      ----------
      column_name : str or list or None
      ddof : int
      method : {'sliding', 'non-overlapping'}
      window_size : int or None

      Returns
      -------
      dict
          {col: {statistics...}, 'metadata': history}



   .. py:method:: cumulative_statistics(column_name=None, method='non-overlapping', window_size=None)

      Generate cumulative mean and uncertainty time series for each column.

      Records operation and returns per-column cumulative arrays plus window_size.



   .. py:method:: additional_data(column_name=None, ddof=1, method='sliding', window_size=None, reduction_factor=0.1)

      Estimate additional sample size needed to reduce SEM by `reduction_factor` via power-law fit.

      Records operation and returns model parameters and sample projections.



   .. py:method:: effective_sample_size_below(column_names=None, alpha=0.05)

      Stub for compatibility with legacy test. Returns dummy value.



   .. py:method:: is_stationary(columns)

      Perform Augmented Dickey-Fuller test for each specified column.

      Records operation in history and returns a dict of bool or error.

      Parameters
      ----------
      columns : str or list of str

      Returns
      -------
      dict
          {column: True if stationary (p<0.05), else False or error message}



   .. py:method:: effective_sample_size(column_names=None, alpha=0.05)

      Compute classic ESS based on significant autocorrelation lags.

      Parameters
      ----------
      column_names : str or list of str or None
          Columns to compute ESS for; defaults to all except 'time'.
      alpha : float
          Significance level for autocorrelation cutoff.

      Returns
      -------
      dict
          {'results': {col: ESS_int or message}}



   .. py:method:: robust_effective_sample_size(x, rank_normalize=True, min_samples=8, return_relative=False)
      :staticmethod:


      Compute a robust ESS via pairwise autocorrelations and optional rank-normalization.

      Parameters
      ----------
      x : array-like
      rank_normalize : bool
      min_samples : int
      return_relative : bool

      Returns
      -------
      float or tuple
          ESS (and ESS/n ratio if return_relative).



   .. py:method:: ess_robust(column_names=None, rank_normalize=False, min_samples=8, return_relative=False)

      Wrapper for `robust_effective_sample_size` over multiple columns.

      Records the operation in history.

      Parameters
      ----------
      column_names : str or list or None
      rank_normalize : bool
      min_samples : int
      return_relative : bool

      Returns
      -------
      dict
          {'results': {col: ESS or tuple}}



   .. py:method:: normalize_data(df)
      :staticmethod:


      Min-Max normalize all signal columns (excluding 'time') to [0,1].

      Parameters
      ----------
      df : pandas.DataFrame

      Returns
      -------
      pandas.DataFrame



