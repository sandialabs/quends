quends.base.ensemble_statistics
===============================

.. py:module:: quends.base.ensemble_statistics

.. autoapi-nested-parse::

   ensemble_statistics.py
   ----------------------
   Module-level statistical functions for ensemble analysis.

   These functions implement the three ensemble-statistics techniques and operate
   on plain lists of :class:`~quends.base.data_stream.DataStream` objects so that
   workflow classes and the :class:`~quends.base.ensemble.Ensemble` class share
   the same computation path without code duplication.

   Technique names
   ---------------
   The three techniques are addressed by descriptive canonical names; legacy
   ``technique=0|1|2`` and ``"technique0"|"technique1"|"technique2"`` strings
   are accepted for backward compatibility but normalised internally.

   * ``"ensemble_average"`` — average traces element-wise into a single trace and
     run DataStream statistics on it (technique 0).
   * ``"pooled_block_means"`` — autotune block window per member until block means
     pass Ljung-Box independence, pool blocks across members, compute statistics
     on the pooled series (technique 1; preferred for trimmed ensembles).
   * ``"ivw_member_means"`` — call ``DataStream.compute_statistics()`` on each
     member, combine the per-member ``(mean, SE)`` pairs using inverse-variance
     weighting (technique 2; fallback to simple mean when SEs are unusable).

   Public API
   ----------
   pool_block_means
   ensemble_average_stats_for_col
   pooled_block_means_stats_for_col
   ivw_member_means_stats_for_col
   compute_ensemble_statistics



Attributes
----------

.. autoapisummary::

   quends.base.ensemble_statistics.ENSEMBLE_AVERAGE
   quends.base.ensemble_statistics.POOLED_BLOCK_MEANS
   quends.base.ensemble_statistics.IVW_MEMBER_MEANS


Functions
---------

.. autoapisummary::

   quends.base.ensemble_statistics.pool_block_means
   quends.base.ensemble_statistics.ensemble_average_stats_for_col
   quends.base.ensemble_statistics.pooled_block_means_stats_for_col
   quends.base.ensemble_statistics.ivw_member_means_stats_for_col
   quends.base.ensemble_statistics.compute_ensemble_statistics


Module Contents
---------------

.. py:data:: ENSEMBLE_AVERAGE
   :value: 'ensemble_average'


.. py:data:: POOLED_BLOCK_MEANS
   :value: 'pooled_block_means'


.. py:data:: IVW_MEMBER_MEANS
   :value: 'ivw_member_means'


.. py:function:: pool_block_means(data_streams, col, window_size = None, method = 'non-overlapping', lb_alpha = 0.05, lb_lags = None, max_tries = 25, min_blocks = 8)

   For all members: autotune per-member window size, then pool block means.

   Calls ``DataStream._process_column`` directly for each member — the single
   autotune entry point — so there is exactly **one** autotune loop per member.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **col** (*str*)
                * **window_size** (*int or None*) -- Starting window hint; auto-estimated via tau_int when ``None``.
                * **method** (*str*)
                * **lb_alpha** (*float*)
                * **lb_lags** (*int or None*) -- Single-lag override for Ljung-Box (backward compat).
                  When ``None``, the canonical ``lag_set=(5, 10)`` is used.
                * **max_tries** (*int*) -- Maximum autotune iterations forwarded to ``autotune_blocks``.
                * **min_blocks** (*int*) -- Hard stop forwarded to ``autotune_blocks``.

   :returns: *(pooled_blocks, meta)*


.. py:function:: ensemble_average_stats_for_col(data_streams, col, ddof = 1, method = 'non-overlapping', window_size = None, avg_ds = None, confidence_level = 0.95, ci_method = 'normal')

   Compute *ensemble_average* (T0) statistics for one column.

   Uses a pre-computed average-ensemble DataStream (or builds it from
   *data_streams* if *avg_ds* is ``None``). Runs
   :meth:`DataStream.compute_statistics` on the single averaged trace.

   :Parameters: * **data_streams** (*list of DataStream*) -- Used to build the average when *avg_ds* is ``None``.
                * **col** (*str*)
                * **ddof** (*int*)
                * **method** (*str*)
                * **window_size** (*int or None*)
                * **avg_ds** (*DataStream or None*) -- Pre-computed ensemble average.

   :returns: *(stat_dict, meta_dict)*


.. py:function:: pooled_block_means_stats_for_col(data_streams, col, ddof = 1, window_size = None, method = 'non-overlapping', lb_lags = None, lb_alpha = 0.05, pooled_lb_alpha_bad = 0.01, confidence_level = 0.95, ci_method = 'normal')

   Compute *pooled_block_means* (T1) statistics for one column.

   Pipeline:

   1. :func:`pool_block_means` calls ``DataStream._process_column`` directly for
      each member → :func:`~quends.base.utils.autotune_blocks` (one autotune per
      member, the single canonical helper). Per-member block ESS and independence
      are recorded there.
   2. ESS and independence are combined **per member** (independent members'
      effective counts add; the ensemble status is all/some/none-independent from
      the per-member verdicts). No test is run on the cross-member concatenation,
      which would inject spurious boundary autocorrelation (AUDIT_REPORT H3).
   3. SEM policy: ``sem_n`` (sd/√n_blocks) when every member is independent;
      otherwise ``sem_ess`` (sd/√ESS_blocks). ``lb_lags`` and
      ``pooled_lb_alpha_bad`` are retained for backward compatibility but no
      longer drive a pooled test.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **col** (*str*)
                * **ddof** (*int*)
                * **window_size** (*int or None*)
                * **method** (*str*)
                * **lb_lags** (*int or None*) -- Single-lag override for the pooled Ljung-Box test (backward compat).
                  When ``None``, the canonical ``lag_set=(5, 10)`` is used.
                * **lb_alpha** (*float*)
                * **pooled_lb_alpha_bad** (*float*)

   :returns: *(stat_dict, meta_dict)* -- ``stat_dict`` canonical keys:

             ``mean``, ``variance``, ``ess_blocks``, ``n_short_averages``,
             ``mean_uncertainty``, ``mean_uncertainty_sem_n``, ``mean_uncertainty_sem_ess``,
             ``se_method``, ``warning``, ``confidence_interval``, ``pm_std``,
             ``window_size`` (median member window, or ``None`` if no members),
             ``independent``, ``independence_status``,
             ``ljungbox_pvalue`` (scalar min, backward-compat),
             ``ljungbox_pvalues`` (list, normalised schema),
             ``ljungbox_lags`` (list, normalised schema),
             ``member_all_independent``, ``member_some_best_p``.


.. py:function:: ivw_member_means_stats_for_col(data_streams, col, ddof = 1, method = 'non-overlapping', window_size = None, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal')

   Compute *ivw_member_means* (T2) statistics for one column.

   Calls :meth:`DataStream.compute_statistics` on each member (which uses the
   canonical :func:`~quends.base.utils.autotune_blocks` helper) and aggregates
   via inverse-variance weighting (fallback: simple mean when no variances are
   available).

   :Parameters: * **data_streams** (*list of DataStream*)
                * **col** (*str*)
                * **ddof** (*int*)
                * **method** (*str*)
                * **window_size** (*int or None*)
                * **diagnostics** (*{"compact", "full"}*) -- ``"full"`` includes per-member statistics (including per-member window
                  and independence diagnostics) in ``stat_dict["individual"]``.

   :returns: *(stat_dict, meta_dict)* -- ``stat_dict`` canonical keys:

             ``mean``, ``mean_uncertainty``, ``confidence_interval``, ``pm_std``,
             ``variance``, ``ess_blocks``, ``n_short_averages``, ``se_method``,
             ``warning``,
             ``window_size`` (mean of per-member windows, or ``None``),
             ``independence_status`` (``"all_independent"``, ``"some_independent"``,
             ``"none_independent"``, or ``"unknown"``),
             ``independent`` (``True`` iff all members are independent),
             ``ljungbox_pvalue`` (mean of per-member min p-values),
             ``ljungbox_pvalues`` (list of per-member min p-values),
             ``individual`` (per-member dicts when *diagnostics* == ``"full"``).


.. py:function:: compute_ensemble_statistics(data_streams, column_name = None, ddof = 1, method = 'non-overlapping', window_size = None, technique = POOLED_BLOCK_MEANS, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal')

   Aggregate mean, SEM, CI, ±SEM, variance, and ESS across an ensemble.

   Dispatches to the appropriate technique helper.

   :Parameters: * **data_streams** (*list of DataStream*)
                * **column_name** (*str, list of str, or None*) -- ``None`` → all common variables.
                * **ddof** (*int*)
                * **method** (*{"non-overlapping", "sliding"}*)
                * **window_size** (*int or None*)
                * **technique** (*{"ensemble_average", "pooled_block_means", "ivw_member_means"}*) -- Canonical technique name.  Legacy values ``0``/``1``/``2`` and
                  ``"technique0"``/``"technique1"``/``"technique2"`` are accepted as
                  backward-compatible aliases.  Default ``"pooled_block_means"``.
                * **diagnostics** (*{"compact", "full"}*)

   :returns: *dict* -- ``{"results": {col: {stats}}, "metadata": {...}}``

             Metadata key for each technique:

             * ``"technique_0_ensemble_average"``
             * ``"technique_1_pooled_block_means"``
             * ``"technique_2_ivw_member_means"``

   :raises ValueError: If *technique* is not a recognised alias.


