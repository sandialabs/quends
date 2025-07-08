quends.postprocessing.plotter
=============================

.. py:module:: quends.postprocessing.plotter


Classes
-------

.. autoapisummary::

   quends.postprocessing.plotter.Plotter


Module Contents
---------------

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



