
Changelog
=========

0.1.6 (2026-11-04)
------------------

**Changed (breaking)**

* ``DataStream.compute_statistics()`` now always computes the standard error
  of the mean as ``sd(block_means) / sqrt(ess_blocks)``, where ``ess_blocks``
  is the Geyer effective sample size of the block means. In 0.1.5 it divided
  by the raw number of blocks whenever the window came from the Ljung-Box
  search, whether or not the test passed. That test has low power for typical
  block counts and often passes while the block means are still correlated,
  so the standard error could be too small. ``mean_uncertainty`` and the
  confidence intervals are therefore larger for almost every call. The means
  do not change.
* The default block window is now ``w = round(tau_int)`` (the "tau-window"
  rule) instead of the Ljung-Box search. This changes ``window_size``,
  ``n_short_averages`` and ``ess_blocks``. ``independence_status`` is now
  ``"tau_window"``. The Ljung-Box results are still reported, but no longer
  affect the window or the standard error.
* ``se_method`` is now always ``"ess_blocks"``. ``"iid_blocks"``,
  ``"iid_blocks_best_p"`` and ``"ess_blocks_fallback"`` are no longer
  returned.
* Ensemble statistics follow the same approach:

  * Pooled block means (T1) always use the ESS-corrected standard error
    (``se_method="sem_ess"``, previously ``"sem_n"`` when all members passed
    Ljung-Box).
  * The ensemble average (T0) and inverse-variance weighted member means (T2)
    by default use a block window based on a correlation time pooled over all
    members (``tau_mode="pooled"``).

* ``MeanVariationTrimStrategy`` now uses the same defaults as
  ``RobustWorkflow``: ``decor_multiplier=6.0`` (was 2.0), ``fudge_fac=0.1``
  (was 1e-6) and ``final_smoothing_window=10`` (was 5). This changes the
  detected SSS start when the strategy is created without these arguments,
  including ``build_trim_strategy("mean_variation")`` and
  ``DataStream.trim(method="mean_variation")``. ``RobustWorkflow`` results do
  not change.
* The regression baselines in ``tests/*/expected/`` were regenerated.

**Added**

* ``DataStream.compute_statistics(window_rule=...)``: ``"tau"`` (default) or
  ``"autotune"`` for the legacy Ljung-Box window search. Both rules use the
  ESS-based standard error. An explicit ``window_size`` overrides the rule.
* Ensemble statistics: ``tau_mode="pooled"|"per_member"`` for T0 and T2 (use
  ``"per_member"`` when members really differ, e.g. parameter scans), and
  ``pooled_ess_method="concat"|"sum"`` for T1. T0 metadata also reports
  ``tau_mode`` and ``tau_star``.
* ``quends.base.utils.tau_window_blocks()`` and
  ``quends.base.utils.pooled_tau_int()`` (pooled correlation time for
  homogeneous ensembles).
* ``quends.MEAN_VARIATION_DEFAULTS``: read-only mapping of the start-of-SSS
  detection defaults, shared by ``MeanVariationTrimStrategy`` and
  ``RobustWorkflow``.

**Fixed**

* Tau-window results no longer get the misleading warning "Too few blocks for
  independence test; SE via Geyer ESS.".
* ``_process_column()`` now rejects an unknown ``window_rule`` even when an
  explicit window is given.

**Documentation**

* New README section "Detecting the Start of Statistical Steady State": when
  to use ``RobustWorkflow``, how the ``mean_variation`` heuristic works, the
  ``verbosity=2`` diagnostic plots, and how to tune its hyperparameters.
* The ``DataStream.compute_statistics()``, ``Ensemble``, ensemble statistics
  and ensemble statistics workflow docstrings now describe the tau-window
  blocking and the ESS-based standard error.

**Security**

* Added ``pillow>=13.0.0`` as a dependency.
* Updated locked dependencies to patched releases: Black 26.10.1,
  certifi 2026.10.23, filelock 4.2.7, Pillow 13.0.0, Tornado 6.6,
  and virtualenv 21.15.2.

0.1.5 (2026-10-15)
------------------

**Changed (breaking)**

* The minimum supported Python version is now 3.12 (previously 3.8).
  Python 3.8 through 3.11 are no longer supported.

**Development**

* Updated the dependency lockfile for Python 3.12 and later.
* Updated the test and documentation CI workflows to run on Python 3.12.

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
