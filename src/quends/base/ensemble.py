from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quends.base.data_stream import DataStream
from quends.base.utils import stationarity_results

# ---------------------------------------------------------------------------
# Reusable helpers (module-level, importable directly by workflow classes)
# ---------------------------------------------------------------------------
from quends.base.ensemble_utils import (
    check_time_steps_uniformity as _check_time_steps_uniformity,
    compute_average_ensemble as _compute_average_ensemble,
    direct_average as _direct_average,
    get_common_variables as _get_common_variables,
    interpolate_to_common_time as _interpolate_to_common_time,
    resolve_cols as _resolve_cols,
    trim_members as _trim_members,
    validate_members,
)
from quends.base.ensemble_statistics import (
    autotune_member_blocks_until_independent as _autotune_blocks,
    compute_ensemble_statistics as _compute_ensemble_statistics,
    pool_block_means as _pool_block_means,
    tech0_stats_for_col as _tech0_stats,
    tech1_pooled_stats_for_col as _tech1_stats,
    tech2_stats_for_col as _tech2_stats,
)

"""
Module: ensemble.py

Three analysis techniques are available (passed as ``technique`` parameter):

  technique=0  Average-ensemble:
    - Construct a single averaged trace (with optional interpolation to a
      common time grid when members have different time axes).
    - Run DataStream statistics on that single averaged trace.
    - This is the standard "average-then-analyze" workflow.

  technique=1  Pooled-block (preferred for trimmed ensembles):
    - For each member and column, autotune window size until block means pass
      Ljung-Box independence, or use the best achievable window.
    - Concatenate per-member block means into one pooled series.
    - Compute statistics on pooled blocks: mean, SEM, CI, ESS.

  technique=2  Member-wise then aggregate:
    - Call DataStream.compute_statistics() on each member.
    - Aggregate using inverse-variance weighting (or simple mean as fallback).

All core statistical and grid logic now lives in the reusable helper modules
``ensemble_utils`` and ``ensemble_statistics``.  The Ensemble class is a thin
wrapper that provides the object-oriented API.
"""


class Ensemble:
    """
    Manages an ensemble of DataStream instances for multi-stream analysis.

    Core logic is implemented in :mod:`quends.base.ensemble_utils` and
    :mod:`quends.base.ensemble_statistics`.  The methods here are thin wrappers
    that delegate to those module-level functions so that workflow classes can
    call the same logic without going through this class.
    """

    def __init__(self, data_streams: List[DataStream]):
        validate_members(data_streams)
        self.data_streams = data_streams

    def __len__(self) -> int:
        return len(self.data_streams)

    def head(self, n: int = 5) -> Dict[int, pd.DataFrame]:
        return {i: ds.head(n) for i, ds in enumerate(self.data_streams)}

    def get_member(self, index: int) -> DataStream:
        return self.data_streams[index]

    def members(self) -> List[DataStream]:
        return self.data_streams

    # ========== Variables & summary ==========

    def common_variables(self) -> List[str]:
        """Column names shared by all members, excluding 'time'."""
        return _get_common_variables(self.data_streams)

    def summary(self) -> Dict:
        info = {}
        for i, ds in enumerate(self.data_streams):
            info[f"Member {i}"] = {
                "n_samples": len(ds.data),
                "columns": list(ds.data.columns),
                "head": ds.head().to_dict(orient="list"),
            }
        out = {
            "n_members": len(self.data_streams),
            "common_variables": self.common_variables(),
            "members": info,
        }
        print(f"Ensemble members: {len(self.data_streams)}")
        print("Common variables:", out["common_variables"])
        return out

    # ========== Helpers ==========

    def _resolve_cols(self, column_name: Optional[Any]) -> List[str]:
        return _resolve_cols(self.data_streams, column_name)

    @staticmethod
    def collect_histories(ds_list: List[DataStream]) -> List:
        return [getattr(ds, "_history", []) for ds in ds_list]

    # ========== Time-grid utilities ==========

    def check_time_steps_uniformity(
        self,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Inspect the time-step regularity of each ensemble member.

        For each member, compute diffs of the 'time' column and classify as:
          - "AllEqual"         all steps identical (within tol)
          - "AllEqualButLast"  all steps equal except the last one
          - "NotUniform"       multiple distinct step sizes

        Returns
        -------
        dict
            {
              "uniform": bool,          True if all members are AllEqual with the same step
              "majority_step": float,   most common step across all members
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

        Notes
        -----
        Delegates to :func:`~quends.base.ensemble_utils.check_time_steps_uniformity`.
        """
        return _check_time_steps_uniformity(
            self.data_streams, tol=tol, verbose=verbose
        )

    def interpolate_to_common_time(
        self,
        method: str = "spline",
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> Tuple["Ensemble", Dict[str, Any]]:
        """
        Interpolate all ensemble members onto a common, regular time grid.

        The common grid spans [min(t_start), max(t_end)] across all members
        using the majority time step.

        Parameters
        ----------
        method : {"spline", "linear"}
            Interpolation method: cubic spline or linear.
        tol : float
            Tolerance for step-size uniformity check.
        verbose : bool
            Print grid diagnostics.

        Returns
        -------
        (interpolated_ensemble, diagnostics_dict)

        Notes
        -----
        Delegates to
        :func:`~quends.base.ensemble_utils.interpolate_to_common_time`.
        """
        new_members, diagnostics = _interpolate_to_common_time(
            self.data_streams, method=method, tol=tol, verbose=verbose
        )
        return self.__class__(new_members), diagnostics

    # ========== Average-ensemble construction ==========

    def compute_average_ensemble(
        self,
        members: Optional[List[DataStream]] = None,
        interp_method: str = "spline",
        tol: float = 1e-8,
        min_coverage: int = 1,
        verbose: bool = False,
    ) -> DataStream:
        """
        Build a single averaged DataStream from ensemble members.

        If all members share the same time grid (detected via
        check_time_steps_uniformity), averages directly.
        If grids differ, interpolates all members to a common grid first.

        Parameters
        ----------
        members : list of DataStream, optional
            Subset to average; defaults to all.
        interp_method : {"spline", "linear"}
            Interpolation method when grids differ.
        tol : float
            Tolerance for uniformity check.
        min_coverage : int
            Minimum number of members that must contribute to a time point.
        verbose : bool
            Print diagnostics.

        Returns
        -------
        DataStream
            Single averaged trace.

        Notes
        -----
        Delegates to
        :func:`~quends.base.ensemble_utils.compute_average_ensemble`.
        """
        data_streams = members if members is not None else self.data_streams
        return _compute_average_ensemble(
            data_streams,
            interp_method=interp_method,
            tol=tol,
            min_coverage=min_coverage,
            verbose=verbose,
        )

    def _direct_average(
        self,
        data_streams: List[DataStream],
        min_coverage: int = 1,
    ) -> Tuple[DataStream, Dict]:
        """Backward-compatible wrapper around :func:`~quends.base.ensemble_utils.direct_average`."""
        cols = self._resolve_cols(None)
        return _direct_average(data_streams, cols=cols, min_coverage=min_coverage)

    # ========== Technique 0: average-ensemble statistics ==========

    def _tech0_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        avg_ds: Optional[DataStream] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute Technique-0 statistics for one column."""
        return _tech0_stats(
            data_streams=self.data_streams,
            col=col,
            ddof=ddof,
            method=method,
            window_size=window_size,
            avg_ds=avg_ds,
        )

    # ========== Technique 1: pooled-block statistics ==========

    def _autotune_member_blocks_until_independent(
        self,
        ds: DataStream,
        col: str,
        window_size: Optional[int],
        method: str = "non-overlapping",
        lb_alpha: float = 0.05,
        lb_lags: Optional[int] = None,
        max_tries: int = 25,
        min_blocks: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Delegate to :func:`~quends.base.ensemble_statistics.autotune_member_blocks_until_independent`."""
        return _autotune_blocks(
            ds=ds,
            col=col,
            window_size=window_size,
            method=method,
            lb_alpha=lb_alpha,
            lb_lags=lb_lags,
            max_tries=max_tries,
            min_blocks=min_blocks,
        )

    def _pooled_block_means_from_trimmed(
        self,
        col: str,
        window_size: Optional[int],
        method: str = "non-overlapping",
        lb_alpha: float = 0.05,
        lb_lags: Optional[int] = None,
        max_tries: int = 25,
        min_blocks: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Delegate to :func:`~quends.base.ensemble_statistics.pool_block_means`."""
        return _pool_block_means(
            data_streams=self.data_streams,
            col=col,
            window_size=window_size,
            method=method,
            lb_alpha=lb_alpha,
            lb_lags=lb_lags,
            max_tries=max_tries,
            min_blocks=min_blocks,
        )

    def _tech1_pooled_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        window_size: Optional[int] = None,
        method: str = "non-overlapping",
        lb_lags: Optional[int] = None,
        lb_alpha: float = 0.05,
        pooled_lb_alpha_bad: float = 0.01,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute Technique-1 (pooled-block) statistics for one column."""
        return _tech1_stats(
            data_streams=self.data_streams,
            col=col,
            ddof=ddof,
            window_size=window_size,
            method=method,
            lb_lags=lb_lags,
            lb_alpha=lb_alpha,
            pooled_lb_alpha_bad=pooled_lb_alpha_bad,
        )

    # ========== Technique 2: member-wise then aggregate ==========

    def _tech2_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        diagnostics: str = "compact",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute Technique-2 statistics for one column."""
        return _tech2_stats(
            data_streams=self.data_streams,
            col=col,
            ddof=ddof,
            method=method,
            window_size=window_size,
            diagnostics=diagnostics,
        )

    # ========== Trim ==========

    def trim(
        self,
        column_name: str,
        method: str = "std",
        window_size: int = 10,
        start_time: float = 0.0,
        threshold: Optional[float] = None,
        robust: bool = True,
        **kwargs,
    ) -> "Ensemble":
        """
        Thin wrapper: trim each ensemble member using the unified trim strategy system.

        Uses :func:`~quends.base.trim.build_trim_strategy` and
        :class:`~quends.base.trim.TrimDataStreamOperation` from ``trim.py``
        directly — the same canonical low-level path used for all trimming.
        No separate trimming logic is defined here.

        Parameters
        ----------
        column_name : str
            Column whose steady-state start drives the trim.
        method : str
            Trim strategy name: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
            ``"self_consistent"``, or ``"iqr"``.
        window_size : int
            Block / rolling-window size (passed through to the strategy).
            Formerly named ``batch_size`` — that name is still accepted as a
            deprecated keyword argument for backward compatibility.
        start_time : float
            Ignore data before this simulation time.
        threshold : float or None
        robust : bool

        Returns
        -------
        Ensemble
            A new Ensemble containing only the members that returned a
            non-empty trimmed DataStream.

        Raises
        ------
        ValueError
            If *method* is unrecognised, or if every member produced an empty
            result (no steady state found in any member).

        Notes
        -----
        Backward compatibility: ``batch_size`` is silently mapped to
        ``window_size`` so that existing callers are not broken.

        Examples
        --------
        >>> trimmed_ens = ens.trim("HeatFlux_st", method="std", window_size=20)
        >>> trimmed_ens = ens.trim("HeatFlux_st", method="iqr", threshold=0.05)
        """
        # Backward-compat: accept batch_size as alias for window_size
        if "batch_size" in kwargs:
            window_size = kwargs.pop("batch_size")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

        kept = _trim_members(
            data_streams=self.data_streams,
            column_name=column_name,
            method=method,
            window_size=window_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )
        if not kept:
            raise ValueError(
                "No ensemble members survived trimming (all failed or returned empty)!"
            )
        return Ensemble(kept)

    # ========== Stationarity ==========

    def is_stationary(self, columns) -> Dict:
        """
        Test stationarity for columns across all members.

        Returns an enriched per-member report (unlike DataStream.is_stationary
        which returns a simple {col: bool} dict).

        Returns
        -------
        dict
            {"results": {"Member i": {col: bool, ...}, ...},
             "metadata": {"Member i": {}, ...}}
        """
        results, meta = {}, {}
        for i, ds in enumerate(self.data_streams):
            r = ds.is_stationary(columns)
            results[f"Member {i}"] = stationarity_results(r)
            meta[f"Member {i}"] = {}
        return {"results": results, "metadata": meta}

    # ========== Mean / SEM / CI ==========

    def mean(
        self,
        column_name=None,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Ensemble mean.

        technique=0  average-ensemble: build single averaged trace, compute stats on it
        technique=1  pooled-block (preferred for trimmed ensembles)
        technique=2  member-wise then inverse-variance aggregate
        """
        cols = self._resolve_cols(column_name)
        result, meta_cols = {}, {}

        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            for col in cols:
                s, m = self._tech0_stats_for_col(col=col, ddof=1, method=method, window_size=window_size, avg_ds=avg_ds)
                result[col] = s.get("mean", np.nan)
                meta_cols[col] = m
            return {"results": result, "metadata": {"technique_0_average_ensemble": meta_cols}}

        elif technique == 1:
            for col in cols:
                s, m = self._tech1_pooled_stats_for_col(col=col, ddof=1, window_size=window_size, method=method)
                result[col] = s.get("mean", np.nan)
                meta_cols[col] = m
            return {"results": result, "metadata": {"technique_1_pooled": meta_cols}}

        elif technique == 2:
            for col in cols:
                s, m = self._tech2_stats_for_col(col=col, ddof=1, method=method, window_size=window_size, diagnostics=diagnostics)
                result[col] = s.get("mean", np.nan)
                meta_cols[col] = m if diagnostics == "full" else {"n_members_used": m.get("n_members_used", 0)}
            return {"results": result, "metadata": {"technique_2_memberwise": meta_cols}}

        else:
            raise ValueError("Invalid technique. Use 0, 1, or 2.")

    def mean_uncertainty(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """Ensemble SEM. See `mean()` for technique semantics."""
        cols = self._resolve_cols(column_name)
        result, meta_cols = {}, {}

        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            for col in cols:
                s, m = self._tech0_stats_for_col(col=col, ddof=ddof, method=method, window_size=window_size, avg_ds=avg_ds)
                result[col] = s.get("mean_uncertainty", np.nan)
                meta_cols[col] = m
            return {"results": result, "metadata": {"technique_0_average_ensemble": meta_cols}}

        elif technique == 1:
            for col in cols:
                s, m = self._tech1_pooled_stats_for_col(col=col, ddof=ddof, window_size=window_size, method=method)
                result[col] = s.get("mean_uncertainty", np.nan)
                meta_cols[col] = m
            return {"results": result, "metadata": {"technique_1_pooled": meta_cols}}

        elif technique == 2:
            for col in cols:
                s, m = self._tech2_stats_for_col(col=col, ddof=ddof, method=method, window_size=window_size, diagnostics=diagnostics)
                result[col] = s.get("mean_uncertainty", np.nan)
                meta_cols[col] = m if diagnostics == "full" else {"n_members_used": m.get("n_members_used", 0)}
            return {"results": result, "metadata": {"technique_2_memberwise": meta_cols}}

        else:
            raise ValueError("Invalid technique. Use 0, 1, or 2.")

    def confidence_interval(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """Ensemble 95% CI. See `mean()` for technique semantics."""
        cols = self._resolve_cols(column_name)
        result, meta_cols = {}, {}

        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            for col in cols:
                s, m = self._tech0_stats_for_col(col=col, ddof=ddof, method=method, window_size=window_size, avg_ds=avg_ds)
                result[col] = s.get("confidence_interval", (np.nan, np.nan))
                meta_cols[col] = m
            return {"results": result, "metadata": {"technique_0_average_ensemble": meta_cols}}

        elif technique == 1:
            for col in cols:
                s, m = self._tech1_pooled_stats_for_col(col=col, ddof=ddof, window_size=window_size, method=method)
                result[col] = s.get("confidence_interval", (np.nan, np.nan))
                meta_cols[col] = m
            return {"results": result, "metadata": {"technique_1_pooled": meta_cols}}

        elif technique == 2:
            for col in cols:
                s, m = self._tech2_stats_for_col(col=col, ddof=ddof, method=method, window_size=window_size, diagnostics=diagnostics)
                result[col] = s.get("confidence_interval", (np.nan, np.nan))
                meta_cols[col] = m if diagnostics == "full" else {"n_members_used": m.get("n_members_used", 0)}
            return {"results": result, "metadata": {"technique_2_memberwise": meta_cols}}

        else:
            raise ValueError("Invalid technique. Use 0, 1, or 2.")

    # ========== Full statistics ==========

    def compute_statistics(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Aggregate mean, SEM, CI, ±SEM, variance, ESS across the ensemble.

        technique=0  Average-ensemble then analyze (average-then-analyze workflow).
        technique=1  Pooled-block (preferred for trimmed ensembles).
        technique=2  Member-wise then aggregate.

        Returns {"results": {col: {stats}}, "metadata": {...}}.

        Notes
        -----
        Delegates to
        :func:`~quends.base.ensemble_statistics.compute_ensemble_statistics`.
        """
        return _compute_ensemble_statistics(
            data_streams=self.data_streams,
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
            technique=technique,
            diagnostics=diagnostics,
        )

    # ========== ESS ==========

    def effective_sample_size(
        self,
        column_names=None,
        alpha: float = 0.05,
        technique: int = 1,
    ) -> Dict:
        """Compute ESS via ensemble statistics (delegates to compute_statistics)."""
        out = self.compute_statistics(column_name=column_names, technique=technique)
        res = {}
        for col, s in out["results"].items():
            if isinstance(s, dict):
                res[col] = s.get("ess_blocks", np.nan)
            else:
                res[col] = np.nan
        return {"results": res, "metadata": out.get("metadata", {})}

    def effective_sample_size_blocks(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
    ) -> Dict:
        """ESS on block means (Geyer) from compute_statistics."""
        out = self.compute_statistics(
            column_name=column_name, ddof=ddof, method=method,
            window_size=window_size, technique=technique,
        )
        res = {col: (s.get("ess_blocks", np.nan) if isinstance(s, dict) else np.nan)
               for col, s in out["results"].items()}
        return {"results": res, "metadata": out.get("metadata", {})}

    def n_short_averages(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
    ) -> Dict:
        """Count of block means from compute_statistics."""
        out = self.compute_statistics(
            column_name=column_name, ddof=ddof, method=method,
            window_size=window_size, technique=technique,
        )
        res = {col: (s.get("n_short_averages", np.nan) if isinstance(s, dict) else np.nan)
               for col, s in out["results"].items()}
        return {"results": res, "metadata": out.get("metadata", {})}
