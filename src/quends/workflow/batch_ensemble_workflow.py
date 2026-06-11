"""
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
"""

from typing import Any, Dict, List, Optional


# Sentinel for "not yet provided"
_UNSET = object()


class BatchEnsembleWorkflow:
    """
    Batch Ensemble workflow — apply a sub-workflow to each ensemble group.

    ``run(groups)`` dispatches every group to the configured sub-workflow and
    returns a structured dict ``{n_items, n_success, results, errors, metadata}``.
    (Cross-group aggregation via :meth:`_aggregate_results` is not yet implemented.)

    Parameters
    ----------
    ensemble_groups : list or None
        A list of ensemble groups to process.  Each group should be an
        :class:`~quends.base.ensemble.Ensemble` object or a list of
        :class:`~quends.base.data_stream.DataStream` objects.
        Validated at construction time; ``None`` is accepted (groups can be
        passed later to :meth:`run`).
    column_name : str or None
        Column to analyse in each ensemble group.
    sub_workflow_type : {"average", "statistics"}
        Which sub-workflow to apply to each ensemble group.
        ``"average"`` → :class:`~quends.workflow.EnsembleAverageWorkflow`;
        ``"statistics"`` → :class:`~quends.workflow.EnsembleStatisticsWorkflow`.
    technique : {"technique1", "technique2", "both"}
        Technique to use when *sub_workflow_type* is ``"statistics"``.
    batch_config : dict or None
        Arbitrary configuration dict forwarded to the sub-workflow constructor.
        See the sub-workflow class docs for valid keys.
    verbosity : int
        ``0`` — silent; ``>0`` — print progress.

    Attributes
    ----------
    _ensemble_groups : list or None
    _column_name : str or None
    _sub_workflow_type : str
    _technique : str
    _batch_config : dict
    _verbosity : int
    """

    _VALID_SUB_WORKFLOWS = frozenset({"average", "statistics"})
    _VALID_TECHNIQUES = frozenset({"pooled_block_means", "ivw_member_means", "both"})

    def __init__(
        self,
        ensemble_groups: Optional[List[Any]] = None,
        column_name: Optional[str] = None,
        sub_workflow_type: str = "statistics",
        technique: str = "both",
        batch_config: Optional[Dict[str, Any]] = None,
        verbosity: int = 0,
    ) -> None:
        # Validate sub_workflow_type
        if sub_workflow_type not in self._VALID_SUB_WORKFLOWS:
            raise ValueError(
                f"sub_workflow_type must be one of "
                f"{sorted(self._VALID_SUB_WORKFLOWS)!r}; "
                f"got {sub_workflow_type!r}."
            )
        # Validate technique — accept canonical names + legacy aliases for BC.
        # The actual normalisation happens in EnsembleStatisticsWorkflow.
        from .ensemble_statistics_workflow import (  # noqa: PLC0415
            _normalize_workflow_technique,
        )
        try:
            technique = _normalize_workflow_technique(technique)
        except ValueError as exc:
            raise ValueError(
                f"technique must be one of {sorted(self._VALID_TECHNIQUES)!r} "
                f"(or legacy 'technique1'/'technique2'); got {technique!r}."
            ) from exc
        # Validate ensemble_groups shape (if provided)
        if ensemble_groups is not None:
            if not isinstance(ensemble_groups, list):
                raise TypeError(
                    "ensemble_groups must be a list of Ensemble or DataStream "
                    f"lists; got {type(ensemble_groups).__name__!r}."
                )
            if len(ensemble_groups) == 0:
                raise ValueError("ensemble_groups must be a non-empty list.")

        self._ensemble_groups = ensemble_groups
        self._column_name = column_name
        self._sub_workflow_type = sub_workflow_type
        self._technique = technique
        self._batch_config = dict(batch_config or {})
        self._verbosity = verbosity

    # ------------------------------------------------------------------
    # Scaffold properties
    # ------------------------------------------------------------------

    @property
    def n_groups(self) -> Optional[int]:
        """Number of ensemble groups, or ``None`` if not yet provided."""
        return len(self._ensemble_groups) if self._ensemble_groups is not None else None

    @property
    def sub_workflow_type(self) -> str:
        return self._sub_workflow_type

    @property
    def technique(self) -> str:
        return self._technique

    # ------------------------------------------------------------------
    # Main entry point (scaffold)
    # ------------------------------------------------------------------

    def run(
        self,
        ensemble_groups: Optional[List[Any]] = None,
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
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

        Parameters
        ----------
        ensemble_groups : list or None
            Override the groups provided at construction time.
        continue_on_error : bool
            If ``True`` (default), catch per-item exceptions and record them
            in the ``"errors"`` map; if ``False``, re-raise the first failure.

        Returns
        -------
        dict
            ``{"workflow": "batch_ensemble", "n_items": int, "n_success": int,
            "n_failed": int, "results": {id: per-item result}, "errors":
            {id: error message}, "metadata": {…}}``

        Raises
        ------
        ValueError
            If no groups are available, or if a per-item dict is malformed.
        TypeError
            If a batch item is not an Ensemble, list, or dict.

        Notes
        -----
        File-based batch loading (CSV directories, NetCDF globs, …) is future
        work; this method only orchestrates already-constructed objects.
        """
        # Local imports to avoid circular dependencies at module load time.
        from .ensemble_average_workflow import EnsembleAverageWorkflow  # noqa: PLC0415
        from .ensemble_statistics_workflow import EnsembleStatisticsWorkflow  # noqa: PLC0415

        groups = ensemble_groups if ensemble_groups is not None else self._ensemble_groups
        if groups is None:
            raise ValueError(
                "No ensemble_groups provided.  Pass them at construction time "
                "or as an argument to run()."
            )
        if not isinstance(groups, list) or len(groups) == 0:
            raise ValueError("ensemble_groups must be a non-empty list.")

        if self._verbosity > 0:
            print(
                f"[BatchEnsembleWorkflow] Starting — {len(groups)} groups, "
                f"default sub_workflow_type='{self._sub_workflow_type}', "
                f"technique='{self._technique}'."
            )

        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for index, item in enumerate(groups):
            # Tentative id for error reporting if unpacking fails.
            item_id_fallback = (
                str(item.get("name", item.get("id", f"item_{index}")))
                if isinstance(item, dict)
                else f"item_{index}"
            )
            try:
                item_id, members, item_workflow_type, item_config = (
                    self._unpack_item(item, index)
                )
                if self._verbosity > 0:
                    print(f"  [{item_id}] running {item_workflow_type} workflow…")
                sub_workflow = self._build_sub_workflow(
                    workflow_type=item_workflow_type,
                    extra_config=item_config,
                )
                results[item_id] = sub_workflow.run(members)
            except Exception as exc:  # noqa: BLE001
                if not continue_on_error:
                    raise
                errors[item_id_fallback] = f"{type(exc).__name__}: {exc}"
                if self._verbosity > 0:
                    print(f"  [{item_id_fallback}] FAILED: {errors[item_id_fallback]}")

        if self._verbosity > 0:
            print(
                f"[BatchEnsembleWorkflow] Done — "
                f"{len(results)} ok, {len(errors)} failed."
            )

        return {
            "workflow": "batch_ensemble",
            "n_items": len(groups),
            "n_success": len(results),
            "n_failed": len(errors),
            "results": results,
            "errors": errors,
            "metadata": {
                "sub_workflow_type": self._sub_workflow_type,
                "technique": self._technique,
                "batch_config": dict(self._batch_config),
                "continue_on_error": continue_on_error,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_workflow_type(value: str) -> str:
        """Map common aliases to the canonical sub-workflow type string."""
        if value in ("average", "ensemble_average", "technique0", "t0", "0"):
            return "average"
        if value in (
            "statistics",
            "ensemble_statistics",
            "techniques",
            "t1_t2",
            "t1",
            "t2",
        ):
            return "statistics"
        raise ValueError(
            f"Unknown workflow_type {value!r}; expected 'average' or 'statistics'."
        )

    def _unpack_item(self, item: Any, index: int):
        """Resolve a batch entry to (item_id, members, workflow_type, extra_config)."""
        # Local import — avoids circular import at module load.
        from ..base.ensemble import Ensemble  # noqa: PLC0415
        from ..base.data_stream import DataStream  # noqa: PLC0415

        item_workflow_type = self._normalize_workflow_type(self._sub_workflow_type)
        item_config: Dict[str, Any] = {}

        if isinstance(item, Ensemble):
            members = item.members()
            item_id = f"item_{index}"
        elif isinstance(item, list):
            if not item or not all(isinstance(d, DataStream) for d in item):
                raise TypeError(
                    f"Batch item {index} is a list, but not all elements are "
                    "DataStream instances."
                )
            members = item
            item_id = f"item_{index}"
        elif isinstance(item, dict):
            payload = item.get("ensemble", item.get("members"))
            if payload is None:
                raise ValueError(
                    f"Batch item {index} is a dict but is missing 'members' or "
                    "'ensemble' key."
                )
            if isinstance(payload, Ensemble):
                members = payload.members()
            elif isinstance(payload, list):
                if not payload or not all(isinstance(d, DataStream) for d in payload):
                    raise TypeError(
                        f"Batch item {index} 'members' must be a non-empty list of "
                        "DataStream instances."
                    )
                members = payload
            else:
                raise TypeError(
                    f"Batch item {index} 'members'/'ensemble' has unsupported "
                    f"type {type(payload).__name__!r}."
                )
            item_id = str(item.get("name", item.get("id", f"item_{index}")))
            if "workflow_type" in item:
                item_workflow_type = self._normalize_workflow_type(
                    item["workflow_type"]
                )
            item_config = dict(item.get("workflow_config", {}) or {})
        else:
            raise TypeError(
                f"Batch item {index} has unsupported type "
                f"{type(item).__name__!r}; expected Ensemble, list, or dict."
            )

        return item_id, members, item_workflow_type, item_config

    def _build_sub_workflow(
        self,
        workflow_type: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Instantiate the configured sub-workflow class with merged config."""
        from .ensemble_average_workflow import EnsembleAverageWorkflow  # noqa: PLC0415
        from .ensemble_statistics_workflow import EnsembleStatisticsWorkflow  # noqa: PLC0415

        wf_type = workflow_type or self._normalize_workflow_type(self._sub_workflow_type)
        merged_config: Dict[str, Any] = dict(self._batch_config)
        if extra_config:
            merged_config.update(extra_config)
        # Forward the configured verbosity unless the per-item config overrides it.
        merged_config.setdefault("verbosity", self._verbosity)

        if wf_type == "average":
            # EnsembleAverageWorkflow requires column_name positionally.
            column = merged_config.pop("column_name", self._column_name)
            if column is None:
                raise ValueError(
                    "EnsembleAverageWorkflow requires 'column_name' — set it "
                    "in the BatchEnsembleWorkflow config or per-item config."
                )
            return EnsembleAverageWorkflow(column_name=column, **merged_config)

        if wf_type == "statistics":
            merged_config.setdefault("technique", self._technique)
            merged_config.setdefault("column_name", self._column_name)
            return EnsembleStatisticsWorkflow(**merged_config)

        raise ValueError(f"Unknown workflow_type {wf_type!r}.")

    def _aggregate_results(self, per_group_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate per-group result dicts into an ensemble-level summary.

        TODO: future work — cross-group mean/std/CI tables.
        """
        raise NotImplementedError("_aggregate_results is not yet implemented.")
