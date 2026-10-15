quends.workflow.batch_ensemble_workflow
=======================================

.. py:module:: quends.workflow.batch_ensemble_workflow

.. autoapi-nested-parse::

   batch_ensemble_workflow.py
   --------------------------
   Batch Ensemble Workflow — run a sub-workflow over many ensemble groups.

   Processes a list of ensemble groups (e.g. multiple parameter configurations,
   scan points, or output directories), dispatching each to a configurable
   sub-workflow (:class:`~quends.workflow.EnsembleAverageWorkflow` or
   :class:`~quends.workflow.EnsembleStatisticsWorkflow`) and collecting the
   per-group results. ``run()`` is implemented and returns a structured dict of
   results and per-group errors; with ``continue_on_error=True`` a failing group
   is logged and skipped rather than aborting the batch.

   Not yet implemented: cross-group aggregation (:meth:`_aggregate_results`),
   parallel execution, and file-discovery of groups.

   Typical usage
   -------------
   >>> from quends.workflow import BatchEnsembleWorkflow
   >>>
   >>> groups = [ensemble_a, ensemble_b, ensemble_c]   # list of Ensemble objects
   >>> workflow = BatchEnsembleWorkflow(
   ...     column_name="HeatFlux_st",
   ...     sub_workflow_type="statistics",
   ...     technique="both",
   ...     verbosity=1,
   ... )
   >>> result = workflow.run(groups)
   >>> result["n_success"], result["n_items"]



Classes
-------

.. autoapisummary::

   quends.workflow.batch_ensemble_workflow.BatchEnsembleWorkflow


Module Contents
---------------

.. py:class:: BatchEnsembleWorkflow(ensemble_groups = None, column_name = None, sub_workflow_type = 'statistics', technique = 'both', batch_config = None, verbosity = 0)

   Batch Ensemble workflow — apply a sub-workflow to each ensemble group.

   ``run(groups)`` dispatches every group to the configured sub-workflow and
   returns a structured dict ``{n_items, n_success, results, errors, metadata}``.
   (Cross-group aggregation via :meth:`_aggregate_results` is not yet implemented.)

   :Parameters: * **ensemble_groups** (*list or None*) -- A list of ensemble groups to process.  Each group should be an
                  :class:`~quends.base.ensemble.Ensemble` object or a list of
                  :class:`~quends.base.data_stream.DataStream` objects.
                  Validated at construction time; ``None`` is accepted (groups can be
                  passed later to :meth:`run`).
                * **column_name** (*str or None*) -- Column to analyse in each ensemble group.
                * **sub_workflow_type** (*{"average", "statistics"}*) -- Which sub-workflow to apply to each ensemble group.
                  ``"average"`` → :class:`~quends.workflow.EnsembleAverageWorkflow`;
                  ``"statistics"`` → :class:`~quends.workflow.EnsembleStatisticsWorkflow`.
                * **technique** (*{"technique1", "technique2", "both"}*) -- Technique to use when *sub_workflow_type* is ``"statistics"``.
                * **batch_config** (*dict or None*) -- Arbitrary configuration dict forwarded to the sub-workflow constructor.
                  See the sub-workflow class docs for valid keys.
                * **verbosity** (*int*) -- ``0`` — silent; ``>0`` — print progress.

   .. attribute:: _ensemble_groups



      :type: list or None

   .. attribute:: _column_name



      :type: str or None

   .. attribute:: _sub_workflow_type



      :type: str

   .. attribute:: _technique



      :type: str

   .. attribute:: _batch_config



      :type: dict

   .. attribute:: _verbosity



      :type: int


   .. py:property:: n_groups
      :type: Optional[int]


      Number of ensemble groups, or ``None`` if not yet provided.


   .. py:property:: sub_workflow_type
      :type: str



   .. py:property:: technique
      :type: str



   .. py:method:: run(ensemble_groups = None, continue_on_error = True)

      Execute the batch workflow over all ensemble groups.

      Each group is dispatched to the appropriate sub-workflow
      (:class:`~quends.workflow.EnsembleAverageWorkflow` for ``"average"``,
      :class:`~quends.workflow.EnsembleStatisticsWorkflow` for
      ``"statistics"``).  This is an in-process, object-level orchestrator —
      file discovery and CSV/NetCDF loading are explicitly out of scope and
      remain future work.

      Each batch item may be:

      * an :class:`~quends.base.ensemble.Ensemble` instance,
      * a list of :class:`~quends.base.data_stream.DataStream` members,
      * a ``dict`` with keys:

        - ``"members"`` *or* ``"ensemble"`` — required, an Ensemble or list of
          DataStreams,
        - ``"name"`` or ``"id"`` — optional human-readable identifier,
        - ``"workflow_type"`` — optional per-item override (``"average"`` or
          ``"statistics"``),
        - ``"workflow_config"`` — optional per-item config dict merged on top
          of ``self._batch_config``.

      :Parameters: * **ensemble_groups** (*list or None*) -- Override the groups provided at construction time.
                   * **continue_on_error** (*bool*) -- If ``True`` (default), catch per-item exceptions and record them
                     in the ``"errors"`` map; if ``False``, re-raise the first failure.

      :returns: *dict* --

                Mapping with the structure::

                    {"workflow": "batch_ensemble", "n_items": int, "n_success": int,
                     "n_failed": int, "results": {id: per-item result}, "errors":
                     {id: error message}, "metadata": {...}}

      :raises ValueError: If no groups are available, or if a per-item dict is malformed.
      :raises TypeError: If a batch item is not an Ensemble, list, or dict.

      .. rubric:: Notes

      File-based batch loading (CSV directories, NetCDF globs, …) is future
      work; this method only orchestrates already-constructed objects.



