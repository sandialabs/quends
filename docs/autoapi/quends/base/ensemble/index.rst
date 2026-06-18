quends.base.ensemble
====================

.. py:module:: quends.base.ensemble


Classes
-------

.. autoapisummary::

   quends.base.ensemble.Ensemble


Module Contents
---------------

.. py:class:: Ensemble(data_streams)

   Manages an ensemble of DataStream instances for multi-stream analysis.

   Core logic is implemented in :mod:`quends.base.ensemble_utils` and
   :mod:`quends.base.ensemble_statistics`.  The methods here are thin wrappers
   that delegate to those module-level functions so that workflow classes can
   call the same logic without going through this class.


   .. py:attribute:: data_streams


   .. py:method:: from_files(paths, variable, *, loader=None)
      :classmethod:


      Build an Ensemble by loading one variable from each file.

      Convenience constructor that replaces the common
      ``[from_csv(p, var) for p in paths]`` boilerplate. ``.nc`` files are
      loaded with :func:`quends.preprocessing.from_netcdf`, everything else
      with :func:`quends.preprocessing.from_csv`, unless an explicit ``loader``
      callable ``loader(path, variable) -> DataStream`` is given.

      :Parameters: * **paths** (*iterable of str or pathlib.Path*) -- File paths to load (one ensemble member each).
                   * **variable** (*str*) -- The variable to load from every file.
                   * **loader** (*callable, optional*) -- Override the auto-selected loader.

      :returns: *Ensemble*



   .. py:method:: head(n = 5)


   .. py:method:: get_member(index)


   .. py:method:: members()


   .. py:method:: common_variables()

      Column names shared by all members, excluding 'time'.



   .. py:method:: summary(verbose = False)

      Return a summary dict of the ensemble.

      :Parameters: **verbose** (*bool*) -- If True, also print a short header. Default False — this is a pure
                   getter and should not print as a side effect.



   .. py:method:: collect_histories(ds_list)
      :staticmethod:



   .. py:method:: check_time_steps_uniformity(tol = 1e-08, verbose = False)

      Inspect the time-step regularity of each ensemble member.

      For each member, compute diffs of the 'time' column and classify as:
        - "AllEqual"         all steps identical (within tol)
        - "AllEqualButLast"  all steps equal except the last one
        - "NotUniform"       multiple distinct step sizes

      :returns: *dict* --

                Mapping with the structure::

                    {
                      "uniform": bool,          # all members AllEqual with the same step
                      "majority_step": float,   # most common step across all members
                      "members": {
                        "Member i": {
                          "status": str,
                          "unique_steps": list[float],
                          "n_steps": int,
                          "t_min": float,
                          "t_max": float,
                        }
                      }
                    }

      .. rubric:: Notes

      Delegates to :func:`~quends.base.ensemble_utils.check_time_steps_uniformity`.



   .. py:method:: interpolate_to_common_time(method = 'spline', tol = 1e-08, verbose = False)

      Interpolate all ensemble members onto a common, regular time grid.

      The common grid spans [min(t_start), max(t_end)] across all members
      using the majority time step.

      :Parameters: * **method** (*{"spline", "linear"}*) -- Interpolation method: cubic spline or linear.
                   * **tol** (*float*) -- Tolerance for step-size uniformity check.
                   * **verbose** (*bool*) -- Print grid diagnostics.

      :returns: *(interpolated_ensemble, diagnostics_dict)*

      .. rubric:: Notes

      Delegates to
      :func:`~quends.base.ensemble_utils.interpolate_to_common_time`.



   .. py:method:: compute_average_ensemble(members = None, interp_method = 'spline', tol = 1e-08, min_coverage = 1, verbose = False)

      Build a single averaged DataStream from ensemble members.

      If all members share the same time grid (detected via
      check_time_steps_uniformity), averages directly.
      If grids differ, interpolates all members to a common grid first.

      :Parameters: * **members** (*list of DataStream, optional*) -- Subset to average; defaults to all.
                   * **interp_method** (*{"spline", "linear"}*) -- Interpolation method when grids differ.
                   * **tol** (*float*) -- Tolerance for uniformity check.
                   * **min_coverage** (*int*) -- Minimum number of members that must contribute to a time point.
                   * **verbose** (*bool*) -- Print diagnostics.

      :returns: *DataStream* -- Single averaged trace.

      .. rubric:: Notes

      Delegates to
      :func:`~quends.base.ensemble_utils.compute_average_ensemble`.



   .. py:method:: trim(column_name, method = 'std', window_size = 10, start_time = 0.0, threshold = None, robust = True, **kwargs)

      Thin wrapper: trim each ensemble member using the unified trim strategy system.

      Uses :func:`~quends.base.trim.build_trim_strategy` and
      :class:`~quends.base.trim.TrimDataStreamOperation` from ``trim.py``
      directly — the same canonical low-level path used for all trimming.
      No separate trimming logic is defined here.

      :Parameters: * **column_name** (*str*) -- Column whose steady-state start drives the trim.
                   * **method** (*str*) -- Trim strategy name: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
                     ``"self_consistent"``, or ``"iqr"``.
                   * **window_size** (*int*) -- Block / rolling-window size (passed through to the strategy).
                     Formerly named ``batch_size`` — that name is still accepted as a
                     deprecated keyword argument for backward compatibility.
                   * **start_time** (*float*) -- Ignore data before this simulation time.
                   * **threshold** (*float or None*)
                   * **robust** (*bool*)

      :returns: *Ensemble* -- A new Ensemble containing only the members that returned a
                non-empty trimmed DataStream.

      :raises ValueError: If *method* is unrecognised, or if every member produced an empty
          result (no steady state found in any member).

      .. rubric:: Notes

      Backward compatibility: ``batch_size`` is silently mapped to
      ``window_size`` so that existing callers are not broken.

      .. rubric:: Examples

      >>> trimmed_ens = ens.trim("HeatFlux_st", method="std", window_size=20)
      >>> trimmed_ens = ens.trim("HeatFlux_st", method="iqr", threshold=0.05)



   .. py:method:: is_stationary(columns)

      Test stationarity for columns across all members.

      Returns an enriched per-member report (unlike DataStream.is_stationary
      which returns a simple {col: bool} dict).

      :returns: *dict* --

                {"results": {"Member i": {col: bool, ...}, ...},
                 "metadata": {"Member i": {}, ...}}



   .. py:method:: mean(column_name=None, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact')

      Ensemble mean.

      technique
          ``"ensemble_average"``     — build single averaged trace, compute stats on it
          ``"pooled_block_means"``   — preferred for trimmed ensembles
          ``"ivw_member_means"``     — member-wise then inverse-variance aggregate
          (legacy ``0/1/2`` and ``"technique0"/1/2`` strings still accepted)



   .. py:method:: mean_uncertainty(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact')

      Ensemble SEM. See `mean()` for technique semantics.



   .. py:method:: confidence_interval(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact')

      Ensemble confidence interval. See `mean()` for technique semantics.



   .. py:method:: compute_statistics(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal')

      Aggregate mean, SEM, CI, ±SEM, variance, ESS across the ensemble.

      technique=``"ensemble_average"``   Single averaged trace, then DataStream stats.
      technique=``"pooled_block_means"`` Preferred for trimmed ensembles.
      technique=``"ivw_member_means"``   Per-member stats, inverse-variance combined.
      (Legacy ``0``/``1``/``2`` and ``"technique0"``/``"technique1"``/``"technique2"``
      also accepted.)

      Confidence-interval parameters (defaults preserve historical 95 %
      normal CI, multiplier ``1.96``):

      confidence_level : float
          Two-sided confidence level.
      ci_method : {'normal', 't'}
          Quantile family.  ``'t'`` is supported for techniques 0 and 1
          (where dof is well-defined); raises for technique 2.

      Returns {"results": {col: {stats}}, "metadata": {...}}.

      .. rubric:: Notes

      Delegates to
      :func:`~quends.base.ensemble_statistics.compute_ensemble_statistics`.



   .. py:method:: compute_uncertainty(method='pooled_block_means', column_name=None, *, ddof = 1, window_size = None, diagnostics = 'compact', confidence_level = 0.95, ci_method = 'normal')

      Friendly alias for :meth:`compute_statistics` keyed by estimator name.

      ``method`` is the estimator: ``"ensemble_average"`` | ``"pooled_block_means"``
      | ``"ivw"`` (plus the legacy ``technique`` aliases). Equivalent to calling
      ``compute_statistics(..., technique=method)`` — the latter still works.

      Returns the same ``{"results": {...}, "metadata": {...}}`` schema.



   .. py:method:: effective_sample_size(column_names=None, alpha = 0.05, technique=POOLED_BLOCK_MEANS)

      Compute ESS via ensemble statistics (delegates to compute_statistics).



   .. py:method:: effective_sample_size_blocks(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS)

      ESS on block means (Geyer) from compute_statistics.



   .. py:method:: n_short_averages(column_name=None, ddof = 1, method = 'non-overlapping', window_size = None, technique=POOLED_BLOCK_MEANS)

      Count of block means from compute_statistics.



