
Changelog
=========

0.1.4 (2026-10-08)
------------------

**Removed (breaking)**

* The ``max_lag_frac`` and ``autocorr_sig_level`` arguments were removed from
  ``RobustWorkflow`` and ``MeanVariationTrimStrategy``. They no longer had any
  effect on the decorrelation time, which is computed by
  ``DataStream.compute_decorrelation_time()``. Passing either argument now
  raises a ``TypeError``. Remove them from existing calls.

**Changed**

* In ``MeanVariationTrimStrategy``, the smoothing window used to detect the
  start of statistical steady state is now ``decor_multiplier * tau_int``,
  capped at the signal length (and at least 3 points). Before, it was also
  capped at ``max_lag_frac * n_pts``. With the default ``decor_multiplier``
  this can give a wider smoothing window, and so a different (often later)
  SSS start time and different statistics than in 0.1.3.
* With ``verbosity > 1``, the autocorrelation plot now covers
  ``1.5 * tau_int`` lags (capped at the signal length), with a dashed vertical
  line at the decorrelation time and a legend.

**Documentation**

* Added a full parameter description to the ``MeanVariationTrimStrategy``
  docstring, and fixed the ``trim()`` parameter docs.
* Fixed the documented default of ``smoothing_window_correction`` in
  ``RobustWorkflow`` (0.5, not 0.8) and clarified ``decor_multiplier``.
* Removed the obsolete arguments from the tutorial notebooks and the
  ``robustworkflow_advanced_guide.py`` example script.

0.1.3 (2026-10-07)
------------------

**Changed**

* The integrated autocorrelation time (``tau_int``) estimate now extends the
  autocorrelation lag window when needed. The ACF is first computed up to
  ``min(n // 4, 2000)`` lags. If the Geyer positive-pair stopping criterion is
  not reached, the number of lags is doubled repeatedly, up to the full series
  length (``n - 1``).
* As a result, ``tau_int`` can be larger, and the effective sample size
  smaller, than in 0.1.2 for strongly correlated series whose initial lag
  window was too short.

**Fixed**

* ``RobustWorkflow.process_data_stream()`` no longer raises
  ``IndexError: single positional indexer is out-of-bounds`` when
  ``start_time`` is later than the last valid data point. In that case it now
  returns NaN statistics if ``operate_safe=True``. If ``operate_safe=False``,
  it returns an ad-hoc estimate over the tail of the data stream, set by
  ``no_sss_tail_fraction``. Both cases report the new status
  ``"StartTimeBeyondData"``.

**Warnings**

* A ``UserWarning`` saying that ``tau_int`` is potentially under-estimated is
  now raised only when the Geyer criterion is not reached even with the
  full-length ACF. The same warning appears in the ``compute_statistics()``
  result metadata.
* This replaces the previous "decorrelation time is large compared to the max
  lag" warning, which was based on a ``tau_int >= 0.5 * nlags`` rule.

**Documentation**

* Updated the README section on publishing to PyPI to describe the automated
  GitHub Release workflow.
* Synchronized ``__version__`` in ``src/quends/__init__.py`` with the version
  in ``pyproject.toml``.

0.1.2
-----

0.1.1
-----

0.1.0 (2025-02-28)
------------------

* First release on PyPI.
