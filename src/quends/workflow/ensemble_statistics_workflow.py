"""
ensemble_statistics_workflow.py
--------------------------------
Ensemble Statistics Workflow (Technique 1 and/or Technique 2).

This workflow supports two member-wise ensemble analysis techniques:

Technique 1 — Pooled-block:
  1. Trim each ensemble member individually.
  2. For each surviving member, autotune the block window size until block
     means pass the Ljung-Box independence test.
  3. Pool all per-member block means into a single series.
  4. Compute statistics (mean, SEM, CI, ESS) on the pooled series.

Technique 2 — Inverse-variance weighted:
  1. Trim each ensemble member individually.
  2. Compute statistics independently on each trimmed member.
  3. Combine per-member results using inverse-variance weighting
     (fallback: simple mean when all SEs are zero or unavailable).

The workflow can run Technique 1 only, Technique 2 only, or both.

All core statistical logic is delegated to
:mod:`quends.base.ensemble_statistics` and
:mod:`quends.base.ensemble_utils`.

Typical usage
-------------
>>> from quends import Ensemble, from_csv
>>> from quends.workflow import EnsembleStatisticsWorkflow
>>>
>>> members = [from_csv("run_a.csv"), from_csv("run_b.csv"), from_csv("run_c.csv")]
>>> workflow = EnsembleStatisticsWorkflow(
...     column_name="HeatFlux_st",
...     technique="both",
...     trim_method="std",
...     window_size=200,
...     start_time=100.0,
...     verbosity=1,
... )
>>> result = workflow.run(members)
>>> print(result["technique1"]["statistics"])
>>> print(result["technique2"]["statistics"])
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..base.data_stream import DataStream
from ..base.ensemble_statistics import (
    compute_ensemble_statistics,
    tech1_pooled_stats_for_col,
    tech2_stats_for_col,
)
from ..base.ensemble_utils import (
    get_common_variables,
    resolve_cols,
    trim_members,
    validate_members,
)

# Valid technique identifiers
_VALID_TECHNIQUES = frozenset({"technique1", "technique2", "both"})


class EnsembleStatisticsWorkflow:
    """
    Ensemble Statistics workflow supporting Technique 1 and/or Technique 2.

    Parameters
    ----------
    column_name : str or None
        Column to analyse.  ``None`` runs all common variables.
    technique : {"technique1", "technique2", "both"}
        Which technique(s) to run.
    trim_method : str
        Trim strategy applied to each member before analysis:
        ``"std"``, ``"threshold"``, ``"rolling_variance"``,
        ``"self_consistent"``, or ``"iqr"``.
    window_size : int
        Block / rolling-window size for the trim strategy.
    start_time : float
        Ignore data before this simulation time during trimming.
    threshold : float or None
        Threshold parameter for trim strategies that need it.
    robust : bool
        Use MAD-based (robust) estimates in the trim strategy where
        supported.
    trim_strategy : TrimStrategy or None
        Pre-built trim strategy object.  When provided, *trim_method*,
        *window_size*, *start_time*, *threshold*, and *robust* are
        ignored for trimming.
    ddof : int
        Degrees of freedom for variance/SEM computation.
    stats_method : {"non-overlapping", "sliding"}
        Block method for statistics.
    stats_window_size : int or None
        Window size for statistics; auto-selected when ``None``.
    diagnostics : {"compact", "full"}
        ``"full"`` includes per-member statistics in Technique 2 output.
    verbosity : int
        ``0`` — silent; ``>0`` — print key steps.
    keep_trimmed : bool
        Include the list of trimmed DataStreams in the result.
    """

    def __init__(
        self,
        column_name: Optional[Any] = None,
        technique: str = "both",
        trim_method: str = "std",
        window_size: int = 200,
        start_time: float = 0.0,
        threshold: Optional[float] = None,
        robust: bool = True,
        trim_strategy: Any = None,
        ddof: int = 1,
        stats_method: str = "non-overlapping",
        stats_window_size: Optional[int] = None,
        diagnostics: str = "compact",
        verbosity: int = 0,
        keep_trimmed: bool = False,
    ) -> None:
        if technique not in _VALID_TECHNIQUES:
            raise ValueError(
                f"technique must be one of {sorted(_VALID_TECHNIQUES)!r}; "
                f"got {technique!r}."
            )
        self._column_name = column_name
        self._technique = technique
        self._trim_method = trim_method
        self._window_size = window_size
        self._start_time = start_time
        self._threshold = threshold
        self._robust = robust
        self._trim_strategy = trim_strategy
        self._ddof = ddof
        self._stats_method = stats_method
        self._stats_window_size = stats_window_size
        self._diagnostics = diagnostics
        self._verbosity = verbosity
        self._keep_trimmed = keep_trimmed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_members(self, ensemble_or_members: Any) -> List[DataStream]:
        from ..base.ensemble import Ensemble  # noqa: PLC0415

        if isinstance(ensemble_or_members, Ensemble):
            return ensemble_or_members.members()
        if isinstance(ensemble_or_members, list):
            return ensemble_or_members
        raise TypeError(
            "ensemble_or_members must be an Ensemble or a list of DataStream objects; "
            f"got {type(ensemble_or_members).__name__!r}."
        )

    def _trim_all_members(self, members: List[DataStream], cols: List[str]) -> Dict[str, List[DataStream]]:
        """
        Trim members once per distinct column.

        In practice, a single trim column is most common; when multiple columns
        are requested each column independently trims the full member set.
        """
        trimmed_by_col: Dict[str, List[DataStream]] = {}
        for col in cols:
            kept = trim_members(
                data_streams=members,
                column_name=col,
                strategy=self._trim_strategy,
                method=self._trim_method,
                window_size=self._window_size,
                start_time=self._start_time,
                threshold=self._threshold,
                robust=self._robust,
            )
            if not kept:
                raise ValueError(
                    f"No ensemble members survived trimming on column '{col}'. "
                    "Adjust trim parameters."
                )
            trimmed_by_col[col] = kept
        return trimmed_by_col

    def _run_technique1(
        self,
        trimmed_by_col: Dict[str, List[DataStream]],
    ) -> Dict[str, Any]:
        """Run pooled-block statistics (Technique 1) for all columns."""
        stats: Dict[str, Any] = {}
        meta_cols: Dict[str, Any] = {}

        for col, kept in trimmed_by_col.items():
            col_stats, col_meta = tech1_pooled_stats_for_col(
                data_streams=kept,
                col=col,
                ddof=self._ddof,
                window_size=self._stats_window_size,
                method=self._stats_method,
            )
            stats[col] = col_stats
            meta_cols[col] = col_meta

        return {
            "statistics": {"results": stats, "metadata": {"technique_1_pooled": meta_cols}},
            "metadata": {
                "technique": "technique1 (pooled-block)",
                "n_trimmed_per_col": {col: len(kept) for col, kept in trimmed_by_col.items()},
            },
        }

    def _run_technique2(
        self,
        trimmed_by_col: Dict[str, List[DataStream]],
    ) -> Dict[str, Any]:
        """Run inverse-variance weighted statistics (Technique 2) for all columns."""
        stats: Dict[str, Any] = {}
        meta_cols: Dict[str, Any] = {}

        for col, kept in trimmed_by_col.items():
            col_stats, col_meta = tech2_stats_for_col(
                data_streams=kept,
                col=col,
                ddof=self._ddof,
                method=self._stats_method,
                window_size=self._stats_window_size,
                diagnostics=self._diagnostics,
            )
            stats[col] = col_stats
            meta_cols[col] = col_meta

        return {
            "statistics": {"results": stats, "metadata": {"technique_2_memberwise": meta_cols}},
            "metadata": {
                "technique": "technique2 (member-wise inverse-variance weighted)",
                "n_trimmed_per_col": {col: len(kept) for col, kept in trimmed_by_col.items()},
                "diagnostics": self._diagnostics,
            },
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, ensemble_or_members: Any) -> Dict[str, Any]:
        """
        Execute the Ensemble Statistics workflow.

        Parameters
        ----------
        ensemble_or_members : Ensemble or list of DataStream
            Raw (untrimmed) ensemble members.

        Returns
        -------
        dict
            Structure depends on *technique*:

            **technique="technique1"**::

              {
                "workflow": "ensemble_statistics",
                "n_members": int,
                "column_name": …,
                "technique": "technique1",
                "technique1": {
                    "trimmed_members": list or None,
                    "statistics": {"results": {col: {…}}, "metadata": {…}},
                    "metadata": {…},
                },
                "metadata": {…},
              }

            **technique="technique2"** — same shape but ``"technique2"`` key.

            **technique="both"** — both ``"technique1"`` and ``"technique2"``
            keys are present.

        Raises
        ------
        TypeError
            If members are not DataStream instances.
        ValueError
            If the member list is empty, or all members are empty after
            trimming, or the column is missing.
        """
        members = self._resolve_members(ensemble_or_members)
        validate_members(members)
        n_members = len(members)

        # Resolve columns
        cols = resolve_cols(members, self._column_name)
        if not cols:
            raise ValueError(
                "No common variables found across ensemble members. "
                "Check that all members share the requested column."
            )

        if self._verbosity > 0:
            print(
                f"[EnsembleStatisticsWorkflow] Starting — {n_members} members, "
                f"technique='{self._technique}', columns={cols}."
            )

        # ── Trim each member per column ───────────────────────────────
        if self._verbosity > 0:
            print(
                f"  Trimming members (method='{self._trim_method}', "
                f"window={self._window_size}, start_time={self._start_time})."
            )
        trimmed_by_col = self._trim_all_members(members, cols)

        for col, kept in trimmed_by_col.items():
            if self._verbosity > 0:
                print(
                    f"  Column '{col}': {len(kept)}/{n_members} members "
                    "survived trimming."
                )

        # ── Run requested technique(s) ────────────────────────────────
        run_t1 = self._technique in ("technique1", "both")
        run_t2 = self._technique in ("technique2", "both")

        t1_result: Optional[Dict] = None
        t2_result: Optional[Dict] = None

        if run_t1:
            if self._verbosity > 0:
                print("  Running Technique 1 (pooled-block)…")
            t1_result = self._run_technique1(trimmed_by_col)
            if self._keep_trimmed:
                # Attach trimmed members (same across techniques when col list is length 1)
                t1_result["trimmed_members"] = {
                    col: kept for col, kept in trimmed_by_col.items()
                }
            else:
                t1_result["trimmed_members"] = None

        if run_t2:
            if self._verbosity > 0:
                print("  Running Technique 2 (inverse-variance weighted)…")
            t2_result = self._run_technique2(trimmed_by_col)
            if self._keep_trimmed:
                t2_result["trimmed_members"] = {
                    col: kept for col, kept in trimmed_by_col.items()
                }
            else:
                t2_result["trimmed_members"] = None

        # ── Build result ───────────────────────────────────────────────
        result: Dict[str, Any] = {
            "workflow": "ensemble_statistics",
            "n_members": n_members,
            "column_name": self._column_name,
            "technique": self._technique,
            "metadata": {
                "trim_method": self._trim_method
                if self._trim_strategy is None
                else "custom_strategy",
                "window_size": self._window_size,
                "start_time": self._start_time,
                "threshold": self._threshold,
                "robust": self._robust,
                "ddof": self._ddof,
                "stats_method": self._stats_method,
                "stats_window_size": self._stats_window_size,
            },
        }

        if self._technique == "technique1":
            result["technique1"] = t1_result
        elif self._technique == "technique2":
            result["technique2"] = t2_result
        else:  # "both"
            result["technique1"] = t1_result
            result["technique2"] = t2_result

        if self._verbosity > 0:
            print("[EnsembleStatisticsWorkflow] Done.")

        return result
