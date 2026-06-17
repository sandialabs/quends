quends.base.trim
================

.. py:module:: quends.base.trim


Attributes
----------

.. autoapisummary::

   quends.base.trim.StandardDeviationTrimStrategy
   quends.base.trim.ThresholdTrimStrategy
   quends.base.trim.RollingVarianceTrimStrategy
   quends.base.trim.SSSStartTrimStrategy


Classes
-------

.. autoapisummary::

   quends.base.trim.TrimStrategy
   quends.base.trim.QuantileTrimStrategy
   quends.base.trim.NoiseThresholdTrimStrategy
   quends.base.trim.RollingVarianceThresholdTrimStrategy
   quends.base.trim.MeanVariationTrimStrategy
   quends.base.trim.SelfConsistentTrimStrategy
   quends.base.trim.IQRTrimStrategy
   quends.base.trim.TrimDataStreamOperation


Functions
---------

.. autoapisummary::

   quends.base.trim.build_trim_strategy


Module Contents
---------------

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



.. py:class:: QuantileTrimStrategy(window_size = 10, start_time = 0.0, robust = True)

   Bases: :py:obj:`TrimStrategy`


   Trim based on standard-deviation / robust MAD steady-state criteria.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


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


.. py:class:: IQRTrimStrategy(window_size = 10, start_time = 0.0, threshold = 0.05)

   Bases: :py:obj:`TrimStrategy`


   Trim using IQR-based steady-state detection:
   IQR(remaining) <= threshold * abs(median(remaining)).

   :Parameters: * **window_size** (*int*) -- Minimum samples before evaluating (used for loop start only).
                * **threshold** (*float*) -- Fraction of abs(median) that IQR must fall below.


   .. py:property:: method_name
      :type: str


      Return the method name for this strategy.


.. py:data:: StandardDeviationTrimStrategy

.. py:data:: ThresholdTrimStrategy

.. py:data:: RollingVarianceTrimStrategy

.. py:data:: SSSStartTrimStrategy

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


.. py:class:: TrimDataStreamOperation(strategy, operation_name = 'trim')

   Bases: :py:obj:`quends.base.operations.DataStreamOperation`


   Operation that applies a TrimStrategy to a DataStream.


   .. py:property:: strategy
      :type: TrimStrategy



