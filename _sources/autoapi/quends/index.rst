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
   quends.MeanVariationTrimStrategy
   quends.NoiseThresholdTrimStrategy
   quends.QuantileTrimStrategy
   quends.RollingVarianceThresholdTrimStrategy
   quends.TrimDataStreamOperation
   quends.TrimStrategy
   quends.Exporter
   quends.Plotter
   quends.RobustWorkflow


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



   .. py:method:: trim(column_name, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True)


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



.. py:class:: DataStreamOperation(operation_name = None, **kwargs)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:property:: name
      :type: str



.. py:class:: MakeDataStreamStationaryOperation(column, n_pts_orig, *, operate_safe=None, n_pts_min=None, n_pts_frac_min=None, drop_fraction=None, verbosity=None)

   Bases: :py:obj:`quends.base.operations.DataStreamOperation`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: column


   .. py:attribute:: n_pts_orig


   .. py:attribute:: is_stationary
      :value: None



   .. py:attribute:: operate_safe
      :value: None



   .. py:attribute:: n_pts_min
      :value: None



   .. py:attribute:: n_pts_frac_min
      :value: None



   .. py:attribute:: drop_fraction
      :value: None



   .. py:attribute:: verbosity
      :value: None



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

      Parameters
      ----------
      col : str
          The name of the column in `data_stream.data` to analyze for steady state.
      workflow : object
          A configuration/workflow object containing parameters:
          - `_max_lag_frac`: Fraction of data used for autocorrelation lag.
          - `_verbosity`: Integer controlling plot and print output levels.
          - `_autocorr_sig_level`: Significance level for the Z-test on lags.
          - `_decor_multiplier`: Multiplier for the calculated decorrelation length.
          - `_std_dev_frac`: Fraction of standard deviation used for tolerance.
          - `_fudge_fac`: Constant to prevent zero-tolerance in noiseless signals.
          - `_smoothing_window_correction`: Factor to adjust for rolling mean lag.
          - `_final_smoothing_window`: Window size for smoothing the metric curves.

      Returns
      -------
      DataStream
          A new DataStream object containing the DataFrame trimmed to the SSS start.
          Returns an empty DataFrame if no SSS is identified.



.. py:class:: NoiseThresholdTrimStrategy(window_size = 10, start_time = 0.0, threshold = None, robust = True)

   Bases: :py:obj:`TrimStrategy`


   Trim using rolling standard deviation on normalized data.


   .. py:method:: apply(data_stream, column_name, **kwargs)

      Template method that defines the trimming workflow.



   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.



.. py:class:: QuantileTrimStrategy(window_size = 10, start_time = 0.0, robust = True)

   Bases: :py:obj:`TrimStrategy`


   Trim based on sliding standard deviation criteria.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.



.. py:class:: RollingVarianceThresholdTrimStrategy(window_size = 50, start_time = 0.0, robust = True, threshold = 0.1)

   Bases: :py:obj:`TrimStrategy`


   Detect steady-state when rolling variance falls below threshold.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.



.. py:data:: RollingVarianceTrimStrategy

.. py:data:: SSSStartTrimStrategy

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



   .. py:method:: apply(data_stream, column_name, **kwargs)

      Template method that defines the trimming workflow.



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



   .. py:method:: steady_state_automatic_plot(data, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, save=False)

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



   .. py:method:: ensemble_steady_state_automatic_plot(ensemble_obj, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, save=False)

      Plot steady state detection automatically for each ensemble member on a grid.

      For each ensemble member in the Ensemble object, for each variable (if multiple are provided,
      all are overlaid on the same subplot), the method uses DataStream.trim() to estimate the steady
      state start time. If detected, it plots the original signal with:

      - A vertical dashed red line at the estimated steady state start.
      - A horizontal green line at the overall mean (computed from the data after steady state).
      - Shaded regions for ±1, ±2, and ±3 standard deviations.

      If no steady state is detected, it plots the raw signal and prints a message.

      The plots are arranged in a grid with one subplot per ensemble member.

      Args:
          ensemble_obj (Ensemble): An Ensemble instance.
          variables_to_plot (list, optional): List of variable names to plot. If None, all columns (except 'time')
              from the first member are used.
          batch_size (int): Window size for the trim() function.
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



.. py:function:: from_csv(file, variable)

   Load a data stream from a CSV file.

   Args:
       file (str): The path to the CSV file.
       variable (str): The column name to load. Must exist in the CSV file.

   Returns:
       DataStream: A DataStream object containing the single specified column.

   Raises:
       ValueError: If the file does not exist or the column is not found.


.. py:function:: from_dict(data_dict, variable)

   Load a data stream from a dictionary.

   Args:
       data_dict (dict): A dictionary where keys are column names and values are lists or arrays of data.
       variables (list, optional): List of variable names (columns) to include.
                                   If None, all dictionary keys are used.

   Returns:
       DataStream: A DataStream object containing the data from the dictionary.


.. py:function:: from_gx(file, variables=None)

   Load a data stream from GX outputs.


.. py:function:: from_json(file, variable)

   Load a single columnas a data stream from a JSON file.

   Args:
       file (str): The path to the JSON file.
       variable (str): The column name to load. Must exist in the JSON file.

   Returns:
       DataStream: A DataStream object containing the single specified column.


.. py:function:: from_netcdf(file, variable)

   Load specified variables from a NetCDF4 file into a pandas DataFrame,
   ensuring all variables have the same length, and extracting only variables
   that end with '_t' or '_st' from the Diagnostics group.

   Args:
       file (str): Path to the NetCDF4 file.
       variables (list, optional): List of variable names to include.
                                   If None, load all eligible variables.

   Returns:
       DataStream: A DataStream object containing the data as a pandas DataFrame.


.. py:function:: from_numpy(np_array, variable)

   Load a single-column data stream from a 1D NumPy array.

   Args:
       np_array (np.ndarray): A 1D NumPy array.
       variable (str): The column name to assign to the array data.

   Returns:
       DataStream: A DataStream object containing the single specified column.

   Raises:
       ValueError: If the input is not a NumPy array or is not 1D.


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

   Attributes
   ----------
   _drop_fraction: float, fraction of data to drop from the start of the DataStream to see if the shortened
       DataStream is stationary.
   _operate_safe : bool
       If True: process data streams in a safe way insisting on stationarity and a segment
       that is clearly in SSS
       If False: try to get some results even if the data stream is not stationary or there is no
       SSS segment found.
   _verbosity: int, level of verbosity for print statements and plots.
       0. : very few print statements or plots
       > 0: more print statements
       > 1: also show plots of intermediate steps
   _drop_fraction: float, fraction of data to drop from the start of the DataStream to see if the shortened
       DataStream is stationary.
   _n_pts_min: int, minimum number of points to keep in the DataStream when shortening it to check for stationarity.
   _n_pts_frac_min: float, minimum fraction of the original number of points to keep in the DataStream when shortening it
       to check for stationarity.
   _max_lag_frac: float, maximum lag (as a fraction of the number of points in the DataStream) to use when computing
       the autocorrelation function to determine the decorrelation length.
   _autocorr_sig_level: float, significance level to use when determining the decorrelation length from the autocorrelation
       function.
   _decor_multiplier: float, multiplier to apply to the decorrelation length to get the smoothing window size.
   _std_dev_frac: float, fraction of the std dev of the stationary signal to use as tolerance when determining the start
       of SSS.
   _fudge_fac: float, fudge factor to multiply the initial mean of the smoothed signal with before adding it to the std dev
       used to compute the tolerance for determining the start of SSS.
   _smoothing_window_correction: float, correction factor to apply to the smoothing window size when determining the start of SSS.
   _final_smoothing_window: int, smoothing window used to avoid quantities going to zero at end of signal.



   .. py:method:: process_irregular_stream(data_stream, col, start_time=0.0)

      Process a data stream that is not stationary or has no steady state segment

      Parameters
      ----------
      data_stream: DataStream
          The data stream to process.
      col: str
          The column name of the quantity of interest in the data stream.
      start_time: float, optional
          The time after which to consider data for processing. Default is 0.0.

      Returns
      -------
      results_dict: dict
          Dictionary with results for the quantity of interest.




   .. py:method:: process_data_steam(data_stream_orig, col, start_time=0.0)

      Process data_stream and handle exceptions gracefully.
      Return mean value and its statistics


      TODO
      * look at number of effective samples we have. Could be low. Allow user to
      override this if they want minimum # of samples for analysis.

      Parameters
      ----------
      data_stream: DataStream
          The data stream to process.
      col: str
          The column name of the quantity of interest in the data stream.
      start_time: float, optional
          The time after which to consider data for processing. Default is 0.0.

      Returns
      -------
      results_dict: dict
          Dictionary with results for the quantity of interest.



   .. py:method:: plot_signal_basic_stats(data_stream, col, stats=None, label=None)

      NOTE: make this part of visualization class?

      Parameters
      ----------
      data_stream: DataStream
          The data stream to plot
      col: str
          The column name of the quantity to plot in the data stream.
      stats: dict, optional
          Dictionary with statistics returned by process_data_steam(). Default is None.
      label: str, optional
          Label to use in title of graph. Default is None.

      Returns
      -------
      shows a plot of the signal with mean, confidence interval and start of SSS (if stats provided)



