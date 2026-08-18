quends.workflow.ensemble_statistics_workflow
============================================

.. py:module:: quends.workflow.ensemble_statistics_workflow

.. autoapi-nested-parse::

   ensemble_statistics_workflow.py
   --------------------------------
   Ensemble Statistics Workflow.

   This workflow supports two member-wise ensemble analysis techniques:

   ``pooled_block_means`` (T1):
     1. Trim each ensemble member individually.
     2. For each surviving member, autotune the block window size until block
        means pass the Ljung-Box independence test.
     3. Pool all per-member block means into a single series.
     4. Compute statistics (mean, SEM, CI, ESS) on the pooled series.

   ``ivw_member_means`` (T2):
     1. Trim each ensemble member individually.
     2. Compute statistics independently on each trimmed member.
     3. Combine per-member results using inverse-variance weighting
        (fallback: simple mean when all SEs are zero or unavailable).

   The workflow can run either technique alone, or both via ``technique="both"``.
   Legacy values ``"technique1"`` / ``"technique2"`` are accepted as backward-
   compatible aliases.

   All core statistical logic is delegated to
   :mod:`quends.base.ensemble_statistics` and
   :mod:`quends.base.ensemble_utils`.

   Typical usage
   -------------
   >>> from quends import Ensemble, from_csv
   >>> from quends.workflow import EnsembleStatisticsWorkflow
   >>>
   >>> members = [from_csv("run_a.csv", "HeatFlux_st"), from_csv("run_b.csv", "HeatFlux_st"), from_csv("run_c.csv", "HeatFlux_st")]
   >>> workflow = EnsembleStatisticsWorkflow(
   ...     column_name="HeatFlux_st",
   ...     technique="both",
   ...     trim_method="std",
   ...     window_size=200,
   ...     start_time=100.0,
   ...     verbosity=1,
   ... )
   >>> result = workflow.run(members)
   >>> print(result["pooled_block_means"]["statistics"])
   >>> print(result["ivw_member_means"]["statistics"])



Classes
-------

.. autoapisummary::

   quends.workflow.ensemble_statistics_workflow.EnsembleStatisticsWorkflow


Module Contents
---------------

.. py:class:: EnsembleStatisticsWorkflow(column_name = None, technique = 'both', trim_method = 'std', window_size = 200, start_time = 0.0, threshold = None, robust = True, trim_strategy = None, ddof = 1, stats_method = 'non-overlapping', stats_window_size = None, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal', verbosity = 0, keep_trimmed = False)

   Ensemble Statistics workflow supporting Technique 1 and/or Technique 2.

   :Parameters: * **column_name** (*str or None*) -- Column to analyse.  ``None`` runs all common variables.
                * **technique** (*{"technique1", "technique2", "both"}*) -- Which technique(s) to run.
                * **trim_method** (*str*) -- Trim strategy applied to each member before analysis:
                  ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                  ``"self_consistent"``, or ``"iqr"``.
                * **window_size** (*int*) -- Block / rolling-window size for the trim strategy.
                * **start_time** (*float*) -- Ignore data before this simulation time during trimming.
                * **threshold** (*float or None*) -- Threshold parameter for trim strategies that need it.
                * **robust** (*bool*) -- Use MAD-based (robust) estimates in the trim strategy where
                  supported.
                * **trim_strategy** (*TrimStrategy or None*) -- Pre-built trim strategy object.  When provided, *trim_method*,
                  *window_size*, *start_time*, *threshold*, and *robust* are
                  ignored for trimming.
                * **ddof** (*int*) -- Degrees of freedom for variance/SEM computation.
                * **stats_method** (*{"non-overlapping", "sliding"}*) -- Block method for statistics.
                * **stats_window_size** (*int or None*) -- Window size for statistics; auto-selected when ``None``.
                * **diagnostics** (*{"compact", "full"}*) -- ``"full"`` includes per-member statistics in Technique 2 output.
                * **verbosity** (*int*) -- ``0`` — silent; ``>0`` — print key steps.
                * **keep_trimmed** (*bool*) -- Include the list of trimmed DataStreams in the result.


   .. py:method:: run(ensemble_or_members)

      Execute the Ensemble Statistics workflow.

      :Parameters: **ensemble_or_members** (*Ensemble or list of DataStream*) -- Raw (untrimmed) ensemble members.

      :returns: *dict* -- Structure depends on *technique*:

                **technique="pooled_block_means"**::

                  {
                    "workflow": "ensemble_statistics",
                    "n_members": int,
                    "column_name": …,
                    "technique": "pooled_block_means",
                    "pooled_block_means": {
                        "trimmed_members": list or None,
                        "statistics": {"results": {col: {…}}, "metadata": {…}},
                        "metadata": {…},
                    },
                    "metadata": {…},
                  }

                **technique="ivw_member_means"** — same shape but
                ``"ivw_member_means"`` key.

                **technique="both"** — both ``"pooled_block_means"`` and
                ``"ivw_member_means"`` keys are present.

                Legacy ``"technique1"`` / ``"technique2"`` inputs are normalised to
                the canonical names; the result dict uses the canonical keys.

      :raises TypeError: If members are not DataStream instances.
      :raises ValueError: If the member list is empty, or all members are empty after
          trimming, or the column is missing.



