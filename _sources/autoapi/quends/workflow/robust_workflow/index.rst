quends.workflow.robust_workflow
===============================

.. py:module:: quends.workflow.robust_workflow


Classes
-------

.. autoapisummary::

   quends.workflow.robust_workflow.RobustWorkflow


Module Contents
---------------

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



