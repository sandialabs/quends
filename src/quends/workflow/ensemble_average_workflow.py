"""
ensemble_average_workflow.py
----------------------------
Ensemble Average Workflow (Technique 0).

This workflow implements the *average-then-analyze* ensemble technique:

  1. Receive individual ensemble members (an :class:`~quends.base.ensemble.Ensemble`
     or a plain list of :class:`~quends.base.data_stream.DataStream`).
  2. Check whether their time grids differ.
  3. If time grids differ, interpolate all members to a common grid.
  4. Compute one averaged DataStream trace from the (possibly interpolated) members.
  5. Trim the single averaged trace to remove the initial transient.
  6. Compute statistics on the trimmed averaged trace.
  7. Return a structured result dictionary.

The order is intentionally **different** from the member-wise workflows:
members are averaged *before* trimming, so the averaged trace is treated as a
single DataStream for the statistical analysis step.

All core logic is delegated to:
- :mod:`quends.base.ensemble_utils` for time-grid and averaging helpers.
- :class:`~quends.base.trim.TrimDataStreamOperation` for trimming.
- :meth:`~quends.base.data_stream.DataStream.compute_statistics` for statistics.

Typical usage
-------------
>>> from quends import Ensemble, from_csv
>>> from quends.workflow import EnsembleAverageWorkflow
>>>
>>> members = [from_csv("run_a.csv"), from_csv("run_b.csv"), from_csv("run_c.csv")]
>>> workflow = EnsembleAverageWorkflow(
...     column_name="HeatFlux_st",
...     trim_method="std",
...     window_size=200,
...     start_time=100.0,
...     verbosity=1,
... )
>>> result = workflow.run(members)
>>> print(result["statistics"])
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..base.data_stream import DataStream
from ..base.ensemble_utils import (
    check_time_steps_uniformity,
    compute_average_ensemble,
    validate_members,
)
from ..base.trim import TrimDataStreamOperation, build_trim_strategy


class EnsembleAverageWorkflow:
    """
    Ensemble Average workflow.

    Averages all ensemble members into a single trace, then trims and analyses
    that averaged trace.  This is the *Technique-0 / average-then-analyze*
    approach.

    Parameters
    ----------
    column_name : str
        Column to analyse.
    trim_method : str
        Trim strategy: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
        ``"self_consistent"``, or ``"iqr"``.
    window_size : int
        Block / rolling-window size passed to the trim strategy.
    start_time : float
        Ignore data before this simulation time during trimming.
    threshold : float or None
        Threshold parameter for strategies that need it (e.g.
        ``"threshold"`` or ``"iqr"``).
    robust : bool
        Use MAD-based (robust) estimates when the trim strategy supports it.
    interp_method : {"spline", "linear"}
        Interpolation method used when time grids differ.
    tol : float
        Tolerance for time-step uniformity check.
    min_coverage : int
        Minimum number of non-NaN members contributing to a time-point in
        the average.
    ddof : int
        Degrees of freedom for variance/SEM computation.
    stats_method : {"non-overlapping", "sliding"}
        Block method for ``DataStream.compute_statistics``.
    stats_window_size : int or None
        Window size for statistics computation; auto-selected when ``None``.
    verbosity : int
        ``0`` — silent; ``>0`` — print key steps; ``>1`` — also print
        intermediate diagnostics.
    keep_intermediate : bool
        If ``True``, include the (untrimmed) averaged DataStream in the
        result under ``"averaged_stream"``.

    Attributes
    ----------
    (all parameters stored with ``_`` prefix, e.g. ``_column_name``)
    """

    def __init__(
        self,
        column_name: str,
        trim_method: str = "std",
        window_size: int = 200,
        start_time: float = 0.0,
        threshold: Optional[float] = None,
        robust: bool = True,
        interp_method: str = "spline",
        tol: float = 1e-8,
        min_coverage: int = 1,
        ddof: int = 1,
        stats_method: str = "non-overlapping",
        stats_window_size: Optional[int] = None,
        verbosity: int = 0,
        keep_intermediate: bool = False,
    ) -> None:
        if not isinstance(column_name, str):
            raise TypeError(
                f"column_name must be a string, got {type(column_name).__name__!r}."
            )
        self._column_name = column_name
        self._trim_method = trim_method
        self._window_size = window_size
        self._start_time = start_time
        self._threshold = threshold
        self._robust = robust
        self._interp_method = interp_method
        self._tol = tol
        self._min_coverage = min_coverage
        self._ddof = ddof
        self._stats_method = stats_method
        self._stats_window_size = stats_window_size
        self._verbosity = verbosity
        self._keep_intermediate = keep_intermediate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_members(
        self,
        ensemble_or_members: Any,
    ) -> List[DataStream]:
        """Accept an Ensemble object or a plain list of DataStreams."""
        # Import here to avoid circular import at module load time
        from ..base.ensemble import Ensemble  # noqa: PLC0415

        if isinstance(ensemble_or_members, Ensemble):
            return ensemble_or_members.members()
        if isinstance(ensemble_or_members, list):
            return ensemble_or_members
        raise TypeError(
            "ensemble_or_members must be an Ensemble or a list of DataStream objects; "
            f"got {type(ensemble_or_members).__name__!r}."
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        ensemble_or_members: Any,
    ) -> Dict[str, Any]:
        """
        Execute the Ensemble Average workflow.

        Parameters
        ----------
        ensemble_or_members : Ensemble or list of DataStream
            The raw (untrimmed) ensemble members to analyse.

        Returns
        -------
        dict
            ``{
              "workflow":               "ensemble_average",
              "n_members":              int,
              "column_name":            str,
              "interpolation_required": bool,
              "common_grid":            dict or None,
              "averaged_stream":        DataStream or None,
              "trimmed_stream":         DataStream,
              "statistics":             dict,
              "trim_history":           list,
              "metadata":               dict,
            }``

        Raises
        ------
        TypeError
            If members are not DataStream instances.
        ValueError
            If the member list is empty, the averaged trace is empty after
            trimming, or the column is not present.
        KeyError
            If *column_name* is absent from the averaged trace.
        """
        members = self._resolve_members(ensemble_or_members)
        validate_members(members)
        n_members = len(members)

        if self._verbosity > 0:
            print(
                f"[EnsembleAverageWorkflow] Starting — {n_members} members, "
                f"column='{self._column_name}'"
            )

        # ── Step 1: check time grids ──────────────────────────────────
        step_info = check_time_steps_uniformity(members, tol=self._tol)
        interpolation_required = not step_info["uniform"]
        common_grid_info: Optional[Dict] = None

        if interpolation_required:
            if self._verbosity > 0:
                print(
                    "  Time grids differ; interpolating to common grid "
                    f"(method='{self._interp_method}')."
                )
            common_grid_info = {
                "majority_step": step_info["majority_step"],
                "method": self._interp_method,
            }
        else:
            if self._verbosity > 1:
                print(
                    f"  Time grids are uniform "
                    f"(step={step_info['majority_step']:.4g}); "
                    "no interpolation needed."
                )

        # ── Step 2: compute averaged trace ────────────────────────────
        avg_ds = compute_average_ensemble(
            members,
            interp_method=self._interp_method,
            tol=self._tol,
            min_coverage=self._min_coverage,
            verbose=(self._verbosity > 1),
        )

        if self._column_name not in avg_ds.data.columns:
            raise KeyError(
                f"Column '{self._column_name}' is not present in the averaged "
                f"ensemble trace. Available columns: {list(avg_ds.data.columns)}."
            )

        if self._verbosity > 0:
            print(
                f"  Averaged trace built: {len(avg_ds.data)} time points, "
                f"columns={list(avg_ds.data.columns)}."
            )

        # ── Step 3: trim the averaged trace ───────────────────────────
        strategy = build_trim_strategy(
            method=self._trim_method,
            window_size=self._window_size,
            start_time=self._start_time,
            threshold=self._threshold,
            robust=self._robust,
        )
        trim_op = TrimDataStreamOperation(strategy=strategy)
        trimmed_ds = trim_op(avg_ds, column_name=self._column_name)

        if trimmed_ds is None or trimmed_ds.data.empty:
            raise ValueError(
                "Trimming the averaged ensemble trace produced an empty DataStream. "
                "Try adjusting start_time, window_size, or trim_method."
            )

        if self._verbosity > 0:
            rows_orig = len(avg_ds.data)
            rows_trim = len(trimmed_ds.data)
            t_start = trimmed_ds.data["time"].iloc[0] if "time" in trimmed_ds.data.columns else "n/a"
            print(
                f"  Averaged trace trimmed: {rows_orig} → {rows_trim} rows "
                f"(steady-state start ≈ {t_start})."
            )

        # ── Step 4: compute statistics on trimmed averaged trace ───────
        statistics = trimmed_ds.compute_statistics(
            column_name=self._column_name,
            ddof=self._ddof,
            method=self._stats_method,
            window_size=self._stats_window_size,
        )

        if self._verbosity > 0:
            col_stats = statistics.get(self._column_name, {})
            print(
                f"  Statistics: mean={col_stats.get('mean', np.nan):.4g}, "
                f"uncertainty={col_stats.get('mean_uncertainty', np.nan):.4g}, "
                f"CI={col_stats.get('confidence_interval', (np.nan, np.nan))}."
            )

        # ── Build result ───────────────────────────────────────────────
        result: Dict[str, Any] = {
            "workflow": "ensemble_average",
            "n_members": n_members,
            "column_name": self._column_name,
            "interpolation_required": interpolation_required,
            "common_grid": common_grid_info,
            "averaged_stream": avg_ds if self._keep_intermediate else None,
            "trimmed_stream": trimmed_ds,
            "statistics": statistics,
            "trim_history": list(trimmed_ds.history.entries()),
            "metadata": {
                "trim_method": self._trim_method,
                "window_size": self._window_size,
                "start_time": self._start_time,
                "threshold": self._threshold,
                "robust": self._robust,
                "interp_method": self._interp_method,
                "ddof": self._ddof,
                "stats_method": self._stats_method,
                "stats_window_size": self._stats_window_size,
                "n_averaged_rows": len(avg_ds.data),
                "n_trimmed_rows": len(trimmed_ds.data),
                "step_info": step_info,
            },
        }

        if self._verbosity > 0:
            print("[EnsembleAverageWorkflow] Done.")

        return result
