
Changelog
=========

0.1.6 (2026-11-03)
------------------

**Changed (breaking)**

* ``DataStream.compute_statistics()`` now always computes the standard error
  of the mean as ``sd(block_means) / sqrt(ess_blocks)``, where ``ess_blocks``
  is the Geyer effective sample size of the block means. Before, it divided by
  the raw number of blocks whenever a Ljung-Box test found the block means
  independent. That test has low power for typical block counts, and often
  passes while the block means are still correlated, so the standard error
  could be too small. As a result, ``mean_uncertainty`` and the confidence
  intervals are larger for almost every call. The means do not change.
* ``se_method`` is now always ``"ess_blocks"``. The values ``"iid_blocks"``,
  ``"iid_blocks_best_p"`` and ``"ess_blocks_fallback"`` are no longer
  returned. Update code that checks for them.
* The default block window is now the integrated autocorrelation time,
  ``w = round(tau_int)`` (the "tau-window" rule), instead of the window found
  by the Ljung-Box autotune search. Because the standard error now accounts
  for correlation between blocks, the blocks no longer need to be independent.
  This changes ``window_size``, ``n_short_averages`` and ``ess_blocks``, and
  ``independence_status`` is now ``"tau_window"`` for the default rule. The
  Ljung-Box diagnostics are still reported. An explicit ``window_size`` still
  overrides the rule.
* Ensemble statistics:

  * ``pooled_block_means_stats_for_col`` (pooled block means, T1) now always
    uses the ESS-corrected standard error (``se_method="sem_ess"``). Before,
    it used the plain block-count standard error (``"sem_n"``) when all
    members passed the independence test. The pooled ESS now comes from the
    concatenated pooled block means by default (new argument
    ``pooled_ess_method="concat"``). Use ``"sum"`` to add up the per-member
    ESS, as before.
  * ``ensemble_average_stats_for_col`` (ensemble average, T0) and
    ``ivw_member_means_stats_for_col`` (inverse-variance weighted member means,
    T2) have a new ``tau_mode`` argument. With the default ``"pooled"``, the
    block window comes from a correlation time that is pooled over all
    members, which is much more stable than one estimated from a single short
    trace. Use ``tau_mode="per_member"`` for ensembles whose members really
    differ, such as parameter scans. T0 metadata now also reports ``tau_mode``
    and ``tau_star``.

* The regression baselines in ``tests/*/expected/`` were regenerated for these
  changes.

**Added**

* ``quends.base.utils.tau_window_blocks()`` blocks a series with the
  tau-window rule.
* ``quends.base.utils.pooled_tau_int()`` estimates one integrated
  autocorrelation time for an ensemble from the mean of the members'
  normalized autocorrelation functions. It is meant for homogeneous ensembles.

**Fixed**

* Results that use the default tau-window rule no longer get the misleading
  warning "Too few blocks for independence test; SE via Geyer ESS.". The
  warning for the ``best_p`` status now says that the standard error uses the
  Geyer ESS.

**Documentation**

* Added a "Detecting the Start of Statistical Steady State" section to the
  README. It covers when to use ``RobustWorkflow``, how the ``mean_variation``
  heuristic works (including the ``smoothing_window_correction`` back
  correction), the ``verbosity=2`` diagnostic plots, and how to tune
  ``std_dev_frac``, ``decor_multiplier`` and ``smoothing_window_correction``.

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
