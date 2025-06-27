quends.base.data_stream
=======================

.. py:module:: quends.base.data_stream


Classes
-------

.. autoapisummary::

   quends.base.data_stream.DataStream


Functions
---------

.. autoapisummary::

   quends.base.data_stream.deduplicate_history
   quends.base.data_stream.to_native_types


Module Contents
---------------

.. py:function:: deduplicate_history(history)

   Remove duplicate operations from a history list, keeping only the most recent occurrence of each operation.

   Scans the history of operations (each represented as a dict with at least an 'operation' key)
   from end to start, retaining only the last entry for each unique operation name while preserving
   the overall order of those final occurrences.

   Parameters
   ----------
   history : list of dict
       Each dict must contain:
         - 'operation': str, the name of the operation.
         - additional keys for operation-specific metadata (e.g., 'options').

   Returns
   -------
   list of dict
       A filtered list with only the final entry of each operation, ordered as in the original list.


.. py:function:: to_native_types(obj)

   Recursively convert NumPy scalar and array types in nested structures to native Python types.

   This function walks through dictionaries, lists, tuples, NumPy scalars, and arrays,
   converting them into Python built-ins:

   - NumPy scalar → Python int or float
   - NumPy array  → Python list (recursively)

   Parameters
   ----------
   obj : any
       The object to convert. Supported container types are dict, list, tuple,
       NumPy ndarray/scalar. Other types are returned unchanged.

   Returns
   -------
   any
       A new object mirroring the input structure but with all NumPy data types replaced
       by their native Python equivalents.


.. py:class:: DataStream(df, _history=None)

   A pipeline for time-series and simulation trace analysis with provenance tracking.

   DataStream encapsulates a pandas DataFrame with a required 'time' column and any number of
   signal columns.  All analysis methods record their operation name and options in an internal
   history, and returned results include deduplicated metadata lineage.

   Core features include:
   - Stationarity testing and steady-state trimming via multiple methods.
   - Statistical summaries: means, uncertainties, confidence intervals, and effective sample size (ESS).
   - Robust ESS estimation using rank-based and pairwise correlation techniques.
   - Incremental and cumulative statistics, plus sample-size planning via power-law fits.

   Attributes
   ----------
   df : pandas.DataFrame
       The underlying time-series data, with 'time' as one column.
   _history : list of dict
       Records of all operations performed, including their options.


   .. py:attribute:: df


<<<<<<< HEAD
   .. py:method:: get_metadata()

      Return the deduplicated operation history for this DataStream.
      Returns
      -------
          list of dict
          The deduplicated operation history, with options for each operation.



=======
>>>>>>> 5f4c24e (Update documentation with tutorials)
   .. py:method:: head(n=5)

      Return the first `n` rows of the underlying DataFrame.

      Parameters
      ----------
      n : int, optional
          Number of rows to return. Defaults to 5.

      Returns
      -------
      pandas.DataFrame
          The first `n` rows of the DataFrame.



   .. py:method:: variables()

      List the signal variable (column) names, excluding the 'time' column.

      Returns
      -------
      Index
          ColumnIndex of variable names in `self.df`.



   .. py:method:: trim(column_name, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True)

      Trim the DataStream to its steady-state portion based on a chosen detection method.
<<<<<<< HEAD
      Always returns a DataStream (possibly empty if trim fails), with operation metadata
      and any messages stored in the _history attribute.
=======

      Records the trim operation in history and returns a dict containing:
        - 'results': a new DataStream of trimmed data or None if trimming failed.
        - 'metadata': deduplicated operation lineage.
        - optionally 'message' on failure.
>>>>>>> 5f4c24e (Update documentation with tutorials)

      Parameters
      ----------
      column_name : str
          Name of the signal column to analyze for steady-state.
      batch_size : int, default=10
          Window size for steady-state detection.
      start_time : float, default=0.0
          Earliest time to consider in the analysis.
      method : {'std', 'threshold', 'rolling_variance'}, default='std'
          Detection method:
<<<<<<< HEAD
          - 'std': sliding std-based criteria (requires stationarity).
          - 'threshold': rolling-std threshold (requires `threshold`).
          - 'rolling_variance': comparison to mean variance times `threshold`.
=======
            - 'std': sliding std-based criteria (requires stationarity).
            - 'threshold': rolling-std threshold (requires `threshold`).
            - 'rolling_variance': comparison to mean variance times `threshold`.
>>>>>>> 5f4c24e (Update documentation with tutorials)
      threshold : float or None
          Threshold value for the 'threshold' or 'rolling_variance' methods.
      robust : bool, default=True
          Use median/MAD instead of mean/std for the 'std' method.

      Returns
      -------
<<<<<<< HEAD
      DataStream
          New DataStream containing the trimmed data, or empty if trimming failed.
          Operation metadata and any messages are in the ._history attribute.
=======
      dict
          {
            'results': DataStream or None,
            'metadata': list of dict,
            'message': str (if occurred)
          }
>>>>>>> 5f4c24e (Update documentation with tutorials)



   .. py:method:: find_steady_state_std(data, column_name, window_size=10, robust=True)
      :staticmethod:


      Identify the earliest time point when the signal remains within ±1/2/3σ proportions.

      Parameters
      ----------
      data : DataFrame
          Subset of the original df (must include 'time' and signal column).
      column_name : str
      window_size : int
          Number of samples to evaluate the steady-state criteria.
      robust : bool
          If True, use median and MAD; else mean and std.

      Returns
      -------
      float or None
          Detected start time of steady-state, or None if not found.



   .. py:method:: find_steady_state_rolling_variance(data, column_name, window_size=50, threshold=0.1)
      :staticmethod:


      Detect steady-state when rolling variance falls below a fraction of its mean.

      Parameters
      ----------
      data : DataFrame
      column_name : str
      window_size : int
      threshold : float
          Fraction of mean rolling std below which to consider steady-state.

      Returns
      -------
      float or None
          Time of first below-threshold variance, or None.



   .. py:method:: normalize_data(df)
      :staticmethod:


      Min-Max normalize all signal columns (excluding 'time') to [0,1].

      Parameters
      ----------
      df : pandas.DataFrame

      Returns
      -------
      pandas.DataFrame



   .. py:method:: find_steady_state_threshold(data, column_name, window_size, threshold)
      :staticmethod:


      Use rolling standard deviation on normalized data to detect steady-state.

      Parameters
      ----------
      data : DataFrame
      column_name : str
      window_size : int
      threshold : float
          Std threshold under which to mark steady-state.

      Returns
      -------
      float or None



   .. py:method:: effective_sample_size(column_names=None, alpha=0.05)

      Compute classic ESS based on significant autocorrelation lags.

      Records the operation in history.

      Parameters
      ----------
      column_names : str or list of str or None
          Columns to compute ESS for; defaults to all except 'time'.
      alpha : float
          Significance level for autocorrelation cutoff.

      Returns
      -------
      dict
          {'results': {col: ESS_int or message}, 'metadata': history}



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
          {'results': {col: ESS or tuple}, 'metadata': history}



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



   .. py:method:: mean(column_name=None, method='non-overlapping', window_size=None)

      Legacy wrapper for test compatibility. Returns only mean (not dict).



   .. py:method:: mean_uncertainty(column_name=None, ddof=1, method='non-overlapping', window_size=None)

      Legacy wrapper for test compatibility. Returns only mean_uncertainty (not dict).



   .. py:method:: confidence_interval(column_name=None, ddof=1, method='non-overlapping', window_size=None)

      Legacy wrapper for test compatibility. Returns only CI tuple.



   .. py:method:: optimal_window_size(method='sliding')

      Stub for compatibility. Return a default or best-guess window size.



   .. py:method:: effective_sample_size_below(column_names=None, alpha=0.05)

      Stub for compatibility with legacy test. Returns dummy value.



