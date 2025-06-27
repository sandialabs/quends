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


Classes
-------

.. autoapisummary::

   quends.DataStream
   quends.Ensemble
   quends.Exporter
   quends.Plotter


Functions
---------

.. autoapisummary::

   quends.from_csv
   quends.from_dict
   quends.from_gx
   quends.from_json
   quends.from_netcdf
   quends.from_numpy


Package Contents
----------------

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



.. py:class:: Ensemble(data_streams)

   Manages an ensemble of DataStream instances, enabling multi-stream analysis.

   Provides methods for:
     - Simple accessors (.head, .get_member, .members).
     - Identifying common variables across streams.
     - Generating an average-ensemble stream aligned to the shortest time grid.
     - Applying DataStream methods (mean, uncertainty, CI, ESS) at the ensemble level
       via three techniques: average-ensemble, aggregate-then-statistics, and weighted.
     - Tracking per-stream and ensemble metadata histories for reproducibility.


   .. py:attribute:: data_streams


   .. py:method:: head(n=5)

      Retrieve the first `n` rows from each DataStream member.

      Parameters
      ----------
      n : int
          Number of rows to return per stream.

      Returns
      -------
      Dict[int, pandas.DataFrame]
          Mapping from member index to its DataFrame head.



   .. py:method:: get_member(index)

      Fetch a specific ensemble member by index.

      Parameters
      ----------
      index : int
          Zero-based index of the DataStream in the ensemble.

      Returns
      -------
      DataStream

      Raises
      ------
      IndexError
          If `index` is out of bounds.



   .. py:method:: members()

      List all ensemble members.

      Returns
      -------
      List[DataStream]



   .. py:method:: common_variables()

      Identify variable columns shared by all members, excluding 'time'.

      Returns
      -------
      List[str]



   .. py:method:: summary()

      Print and return a structured summary of ensemble members.

      Includes each member's sample count, column list, and head rows.

      Returns
      -------
      dict
          { 'n_members': int,
            'common_variables': List[str],
            'members': { 'Member i': { 'n_samples': int,
                                        'columns': List[str],
                                        'head': dict } } }



   .. py:method:: compute_average_ensemble(members = None)

      Build a DataStream whose columns are the elementwise mean across members,
      aligned on the shortest time grid.

      Parameters
      ----------
      members : List[DataStream], optional
          Subset of streams to average; defaults to all.

      Returns
      -------
      DataStream

      Raises
      ------
      ValueError
          If no streams are provided.



   .. py:method:: resample_to_short_intervals(short_df, long_df)

      Align `long_df` onto `short_df.time` by block-averaging between boundaries.

      Parameters
      ----------
      short_df : pandas.DataFrame
          Reference DataFrame with the shortest time series.
      long_df : pandas.DataFrame
          Stream to resample.

      Returns
      -------
      pandas.DataFrame
          Resampled data matching `short_df.time`.



   .. py:method:: collect_histories(ds_list)
      :staticmethod:


      Gather `_history` lists from each DataStream in `ds_list`.

      Parameters
      ----------
      ds_list : List[DataStream]
          Streams whose histories to collect.

      Returns
      -------
      List[List[dict]]



<<<<<<< HEAD
   .. py:method:: trim(column_name, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True)
=======
   .. py:method:: trim(column_name, window_size = 10, start_time = 0.0, method = 'std', threshold = None, robust = True)

      Apply steady-state trimming to each member on `column_name`.

      Returns
      -------
      dict
          { 'results': Ensemble or None,
            'metadata': Dict[str, Any] }

>>>>>>> 5f4c24e (Update documentation with tutorials)


   .. py:method:: is_stationary(columns)

      Test stationarity for `columns` across all members.

      Returns
      -------
      dict
          { 'results': {Member i: {col: bool or error}},
            'metadata': {Member i: history} }



   .. py:method:: effective_sample_size(column_names=None, alpha = 0.05, technique = 0)

      Compute classic ESS via three techniques:
        0 - on average-ensemble
        1 - on concatenated aggregate
        2 - per-member then aggregate

      Returns
      -------
      dict
          { 'results': ..., 'metadata': ... }



   .. py:method:: ess_robust(column_names=None, rank_normalize=True, min_samples=8, return_relative=False, technique=0)

      Compute robust ESS (rank-based) via three techniques.

      Returns
      -------
      dict
          { 'results': ..., 'metadata': ... }



   .. py:method:: mean(column_name=None, method='non-overlapping', window_size=None, technique=0)

      Compute ensemble mean via three techniques:
        0 - average-ensemble
        1 - aggregate-then-statistics
        2 - weighted per-member

      Returns
      -------
      dict
          { 'results': ..., 'metadata': ... }



   .. py:method:: mean_uncertainty(column_name=None, ddof=1, method='non-overlapping', window_size=None, technique=0)

      Compute SEM via three techniques (0: average, 1: aggregate, 2: weighted).

      Returns
      -------
      dict



   .. py:method:: confidence_interval(column_name=None, ddof=1, method='non-overlapping', window_size=None, technique=0)

      Compute 95% CI via three techniques.

      Returns
      -------
      dict



   .. py:method:: compute_statistics(column_name=None, ddof=1, method='non-overlapping', window_size=None, technique=0)

      Aggregate mean, SEM, CI, and ±1std across the ensemble.

      Returns
      -------
      dict
          { 'results': {col: {stats}}, 'metadata': {...} }



.. py:class:: Exporter(output_dir='exported_results')

   A class for exporting data/results in various formats: DataFrame, JSON, dictionary, and NumPy array.
   Provides both display (print to console) and save (to file) functions.
   Includes automatic conversion of NumPy types to native Python types for compatibility.


   .. py:attribute:: output_dir
      :value: 'exported_results'



   .. py:method:: to_native_types(obj)
      :staticmethod:


      Recursively convert NumPy scalar types in dicts/lists/tuples to native Python types.
      Compatible with NumPy 2.x (no `np.float_`, `np.int_`, etc.).



   .. py:method:: to_dataframe(data)

      Convert input data to a pandas DataFrame.

      Args:
          data: DataFrame, dict, NumPy array, or any structure convertible to DataFrame.

      Returns:
          pd.DataFrame: The converted DataFrame.



   .. py:method:: to_dictionary(data)

      Convert input data to a dictionary, and make all types native Python.

      Args:
          data: dict, DataFrame, or NumPy array.

      Returns:
          dict: The converted dictionary (native types).



   .. py:method:: to_numpy(data)

      Convert input data to a NumPy array.

      Args:
          data: np.ndarray, DataFrame, or dict.

      Returns:
          np.ndarray: The converted NumPy array.



   .. py:method:: to_json(data)

      Convert input data to a JSON string (with native Python types).

      Args:
          data: DataFrame, dict, or NumPy array.

      Returns:
          str: The JSON string.



   .. py:method:: display_dataframe(data, head=None)

      Display data as a DataFrame.

      Args:
          data: Data convertible to DataFrame.
          head (int, optional): If provided, only display the first 'head' rows.



   .. py:method:: display_dictionary(data)

      Display data as a dictionary, with all native types.

      Args:
          data: Data convertible to dictionary.



   .. py:method:: display_numpy(data)

      Display data as a NumPy array.

      Args:
          data: Data convertible to a NumPy array.



   .. py:method:: display_json(data)

      Display data as a JSON string, with all native types.

      Args:
          data: Data convertible to JSON.



   .. py:method:: save_dataframe(data, file_name='dataframe.csv')

      Save data as a CSV file (DataFrame format).

      Args:
          data: Data convertible to DataFrame.
          file_name (str): Name of the file (default: 'dataframe.csv').



   .. py:method:: save_dictionary(data, file_name='data_dictionary.json')

      Save data as a JSON file representing a dictionary.

      Args:
          data: Data convertible to a dictionary.
          file_name (str): Name of the file (default: 'data_dictionary.json').



   .. py:method:: save_numpy(data, file_name='data.npy')

      Save data as a NumPy array file.

      Args:
          data: Data convertible to a NumPy array.
          file_name (str): Name of the file (default: 'data.npy').



   .. py:method:: save_json(data, file_name='data.json')

      Save data as a JSON file (with all native types).

      Args:
          data: Data convertible to JSON.
          file_name (str): Name of the file (default: 'data.json').



.. py:class:: Plotter(output_dir='results_figures')

   A class that encapsulates plotting functionality for time series data.


   .. py:attribute:: output_dir
      :value: 'results_figures'



   .. py:method:: format_dataset_name(dataset_name)
      :staticmethod:


      Format the dataset name for display and file naming.

      Args:
          dataset_name (str): The original dataset name.

      Returns:
          str: A formatted dataset name.



   .. py:method:: trace_plot(data, variables_to_plot=None, save=False)

      Plot individual (trace) time series data from a DataStream or a dictionary of DataFrames.
      The resulting plots are displayed and optionally saved if 'save' is True.

      Args:
          data (DataStream or dict): A DataStream instance or dictionary of DataFrames.
          variables_to_plot (list, optional): List of variables to plot. If None,
              all columns (except 'time') from the first DataFrame are used.
          save (bool, optional): If True, save the generated plots to the output directory.
                                 Defaults to False.



   .. py:method:: trace_plot_with_mean(data, variables_to_plot=None, save=False)

      Plot individual (trace) time series data from a DataStream or a dictionary of DataFrames.
      The resulting plots are displayed and optionally saved if 'save' is True.

      Args:
          data (DataStream or dict): A DataStream instance or dictionary of DataFrames.
          variables_to_plot (list, optional): List of variables to plot. If None,
              all columns (except 'time') from the first DataFrame are used.
          save (bool, optional): If True, save the generated plots to the output directory.
                                 Defaults to False.



   .. py:method:: ensemble_trace_plot(data, variables_to_plot=None, save=False)

      Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.
      The resulting plots are displayed and optionally saved if 'save' is True.

      Args:
          data (DataStream or dict): A DataStream instance or dictionary of DataFrames representing ensemble members.
          variables_to_plot (list, optional): List of variables to plot. If None,
              all columns (except 'time') from the first DataFrame are used.
          save (bool, optional): If True, save the generated plots to the output directory. Defaults to False.



   .. py:method:: ensemble_trace_plot_with_mean(data, variables_to_plot=None, save=False)

      Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.
      The resulting plots are displayed and optionally saved if 'save' is True.

      Args:
          data (DataStream or dict): A DataStream instance or dictionary of DataFrames representing ensemble members.
          variables_to_plot (list, optional): List of variables to plot. If None,
              all columns (except 'time') from the first DataFrame are used.
          save (bool, optional): If True, save the generated plots to the output directory.
                                 Defaults to False.



<<<<<<< HEAD
   .. py:method:: steady_state_automatic_plot(data, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, save=False)
=======
   .. py:method:: steady_state_automatic_plot(data, variables_to_plot=None, window_size=10, start_time=0.0, method='std', threshold=None, robust=True, save=False)
>>>>>>> 5f4c24e (Update documentation with tutorials)

      Plot steady state detection for each variable in the data. For each variable, the method uses the
      DataStream.trim() function to estimate the steady state start time. If a steady state is detected,
      the function plots the original time series along with:

          - A vertical dashed red line indicating the steady state start time.
          - A horizontal green line at the overall mean (computed from data after the steady state start).
          - Shaded regions representing ±1, ±2, and ±3 standard deviations.

      If no steady state is detected for a variable, the full signal is plotted and a message is printed.

      Args:
          data (DataStream or dict): A DataStream instance or a dictionary of DataFrames.
          variables_to_plot (list, optional): List of variables to plot. If None,
              all columns (except 'time') from the first DataFrame are used.
          window_size (int, optional): Window size to use in the trim() function.
          start_time (float, optional): Start time for steady state detection.
          method (str, optional): Method to use for steady state detection ('std', 'threshold', or 'rolling_variance').
          threshold (float, optional): Threshold value required for 'threshold' or 'rolling_variance' methods.
          robust (bool, optional): Whether to use robust statistics (median/MAD) in the 'std' method.
          save (bool, optional): If True, save the plot to disk. Defaults to False.



   .. py:method:: steady_state_plot(data, variables_to_plot=None, steady_state_start=None, save=False)

      Plot steady state detection for each variable in the data using a user-supplied steady state start.
      The user can provide a single float (applied to all variables) or a dictionary mapping variable names to floats.
      For each variable, if a steady state start is provided, the plot displays:

          - The full signal.
          - A vertical dashed red line at the given steady state start.
          - A horizontal green line for the mean (after steady state).
          - Shaded regions for ±1, ±2, and ±3 standard deviations (after steady state).

      If no steady state start is provided for a variable, only the raw signal is plotted and a message is printed.

      Args:
          data (DataStream or dict): A DataStream instance or dictionary of DataFrames.
          variables_to_plot (list, optional): List of variables to plot. If None,
              all columns (except 'time') from the first DataFrame are used.
          steady_state_start (float or dict, optional): Either a single steady state start (float)
              applied to all variables or a dictionary mapping variable names to steady state start values.
          save (bool): If True, the generated plots are saved to the output directory.



   .. py:method:: plot_acf(data, alpha=0.05, column=None, ax=None)

      Plot the Autocorrelation Function (ACF) for a given data stream or array-like object.

      If 'data' is a DataStream, the function extracts the specified column (or the first column
      that is not "time" if not specified). Otherwise, data is assumed to be 1D array-like.

      The function computes:
        - nlags = int(n / 3), where n is the number of observations.
        - ACF values using statsmodels.tsa.stattools.acf.
        - A 95% confidence interval as: conf_interval = z_critical / sqrt(n),
          where z_critical is computed from the two-tailed test.

      If an axis (ax) is provided, the plot is drawn on that axis; otherwise, a new figure is created.

      Args:
          data (DataStream or array-like): The data to plot.
          alpha (float): Significance level for the confidence interval (default: 0.05).
          column (str, optional): Column name to use if data is a DataStream. Defaults to the first non-'time' column.
          ax (matplotlib.axes.Axes, optional): Axis on which to plot. If None, a new figure is created.



   .. py:method:: plot_acf_ensemble(ensemble_obj, alpha=0.05, column=None)

      Plot the ACF for each ensemble member individually on a grid of subplots.

      The number of rows and columns in the grid is determined based on the number of ensemble members.
      This function loops through each ensemble member (DataStream) in the Ensemble object and calls
      the plot_acf function to generate the individual ACF plots on separate subplots.

      Args:
          ensemble_obj (Ensemble): An Ensemble instance containing DataStream members.
          alpha (float): Significance level for the confidence interval (default: 0.05).
          column (str, optional): Column name to use for ACF computation. If None, the first non-'time' column is used.



<<<<<<< HEAD
   .. py:method:: ensemble_steady_state_automatic_plot(ensemble_obj, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, save=False)
=======
   .. py:method:: ensemble_steady_state_automatic_plot(ensemble_obj, variables_to_plot=None, window_size=10, start_time=0.0, method='std', threshold=None, robust=True, save=False)
>>>>>>> 5f4c24e (Update documentation with tutorials)

      Plot steady state detection automatically for each ensemble member on a grid.

      For each ensemble member in the Ensemble object, for each variable (if multiple are provided,
      all are overlaid on the same subplot), the method uses DataStream.trim() to estimate the steady
      state start time. If detected, it plots the original signal with:

<<<<<<< HEAD
      - A vertical dashed red line at the estimated steady state start.
      - A horizontal green line at the overall mean (computed from the data after steady state).
      - Shaded regions for ±1, ±2, and ±3 standard deviations.
=======
        - A vertical dashed red line at the estimated steady state start.
        - A horizontal green line at the overall mean (computed from the data after steady state).
        - Shaded regions for ±1, ±2, and ±3 standard deviations.
>>>>>>> 5f4c24e (Update documentation with tutorials)

      If no steady state is detected, it plots the raw signal and prints a message.

      The plots are arranged in a grid with one subplot per ensemble member.

      Args:
          ensemble_obj (Ensemble): An Ensemble instance.
          variables_to_plot (list, optional): List of variable names to plot. If None, all columns (except 'time')
              from the first member are used.
<<<<<<< HEAD
          batch_size (int): Window size for the trim() function.
=======
          window_size (int): Window size for the trim() function.
>>>>>>> 5f4c24e (Update documentation with tutorials)
          start_time (float): Start time for steady state detection.
          method (str): Steady state detection method ('std', 'threshold', or 'rolling_variance').
          threshold (float, optional): Threshold if needed by the method.
          robust (bool): If True, use robust statistics (median/MAD) in the 'std' method.
          save (bool): If True, save the resulting figure to disk.

      Returns:
          None



   .. py:method:: ensemble_steady_state_plot(ensemble_obj, variables_to_plot=None, steady_state_start=None, save=False)

      Plot steady state detection for each ensemble member using a user-supplied steady state start.

      For each ensemble member in the Ensemble object, the function plots the signal for the specified
      variables (or all non-'time' variables if not provided) and draws:

        - A vertical dashed red line at the user-supplied steady state start.
        - A horizontal green line representing the mean of the data after the steady state.
        - Shaded regions for ±1, ±2, and ±3 standard deviations (computed after the steady state).

      If no steady state start is provided for a variable, the raw signal is plotted and a message is printed.

      The plots are arranged in a grid.

      Args:
          ensemble_obj (Ensemble): An Ensemble instance.
          variables_to_plot (list, optional): List of variables to plot. If None, all non-'time' columns are used.
          steady_state_start (float or dict, optional): A single float or a dict mapping variable names to steady state start values.
          save (bool): If True, save the resulting figure.



   .. py:method:: plot_ensemble(ensemble_obj, variables_to_plot=None, show_plots=False, save=False)

      Plot each ensemble member together with the ensemble average,
      arranged in 2 columns and as many rows as needed.
      Legend is centered below the grid, with just enough room reserved.



.. py:function:: from_csv(file, variables=None)

   Load a data stream from a CSV file.

   Args:
       file (str): The path to the CSV file.
       variables (list): Variable names (columns) to load (default: None, which loads all columns).

   Returns:
       DataStream: A DataStream object containing the data from the CSV file.


.. py:function:: from_dict(data_dict, variables=None)

   Load a data stream from a dictionary.

   Args:
       data_dict (dict): A dictionary where keys are column names and values are lists or arrays of data.
       variables (list, optional): List of variable names (columns) to include.
                                   If None, all dictionary keys are used.

   Returns:
       DataStream: A DataStream object containing the data from the dictionary.


.. py:function:: from_gx(file, variables=None)

   Load a data stream from GX outputs.


.. py:function:: from_json(file, variables=None)

   Load a data stream from a JSON file.

   Args:
       file (str): The path to the JSON file.
       variables (list, optional): List of variable names (columns) to load.
                                   If None, all columns are loaded.

   Returns:
       DataStream: A DataStream object containing the data from the JSON file.


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


.. py:function:: from_numpy(np_array, variables=None)

   Load a data stream from a NumPy array.

   Args:
       np_array (np.ndarray): A 1D or 2D NumPy array.
       variables (list, optional): List of column names. For a 1D array, a single-column name is used.
                                   For a 2D array, the length of variables must match the number of columns.
                                   If None, default column names are assigned.

   Returns:
       DataStream: A DataStream object containing the NumPy array data.


