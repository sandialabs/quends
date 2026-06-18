from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_stream import DataStream
from .utils import stationarity_results

# ---------------------------------------------------------------------------
# Reusable helpers (module-level, importable directly by workflow classes)
# ---------------------------------------------------------------------------
from .ensemble_utils import (
    check_time_steps_uniformity as _check_time_steps_uniformity,
    compute_average_ensemble as _compute_average_ensemble,
    direct_average as _direct_average,
    get_common_variables as _get_common_variables,
    interpolate_to_common_time as _interpolate_to_common_time,
    resolve_cols as _resolve_cols,
    trim_members as _trim_members,
    validate_members,
)
from .ensemble_statistics import (
    ENSEMBLE_AVERAGE,
    IVW_MEMBER_MEANS,
    POOLED_BLOCK_MEANS,
    _normalize_technique,
    compute_ensemble_statistics as _compute_ensemble_statistics,
    ensemble_average_stats_for_col as _ensemble_average_stats,
    ivw_member_means_stats_for_col as _ivw_member_means_stats,
    pooled_block_means_stats_for_col as _pooled_block_means_stats,
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

    @classmethod
    def from_files(cls, paths, variable: str, *, loader=None) -> "Ensemble":
        """Build an Ensemble by loading one variable from each file.

        Convenience constructor that replaces the common
        ``[from_csv(p, var) for p in paths]`` boilerplate. ``.nc`` files are
        loaded with :func:`quends.preprocessing.from_netcdf`, everything else
        with :func:`quends.preprocessing.from_csv`, unless an explicit ``loader``
        callable ``loader(path, variable) -> DataStream`` is given.

        Parameters
        ----------
        paths : iterable of str or pathlib.Path
            File paths to load (one ensemble member each).
        variable : str
            The variable to load from every file.
        loader : callable, optional
            Override the auto-selected loader.

        Returns
        -------
        Ensemble
        """
        # Local imports avoid importing the preprocessing layer at module load.
        from ..preprocessing.csv import from_csv
        from ..preprocessing.netcdf import from_netcdf

        members: List[DataStream] = []
        for p in paths:
            sp = str(p)
            if loader is not None:
                ds = loader(sp, variable)
            elif sp.endswith(".nc"):
                ds = from_netcdf(sp, variable)
            else:
                ds = from_csv(sp, variable)
            members.append(ds)
        return cls(members)

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

    def summary(self, verbose: bool = False) -> Dict:
        """Return a summary dict of the ensemble.

        Parameters
        ----------
        verbose : bool
            If True, also print a short header. Default False — this is a pure
            getter and should not print as a side effect.
        """
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
        if verbose:
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

    # ========== ensemble_average (T0): single averaged trace ==========

    def _ensemble_average_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        avg_ds: Optional[DataStream] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute ensemble_average (T0) statistics for one column."""
        return _ensemble_average_stats(
            data_streams=self.data_streams,
            col=col,
            ddof=ddof,
            method=method,
            window_size=window_size,
            avg_ds=avg_ds,
        )

    # ========== pooled_block_means (T1) ==========

    def _pooled_block_means_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        window_size: Optional[int] = None,
        method: str = "non-overlapping",
        lb_lags: Optional[int] = None,
        lb_alpha: float = 0.05,
        pooled_lb_alpha_bad: float = 0.01,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute pooled_block_means (T1) statistics for one column."""
        return _pooled_block_means_stats(
            data_streams=self.data_streams,
            col=col,
            ddof=ddof,
            window_size=window_size,
            method=method,
            lb_lags=lb_lags,
            lb_alpha=lb_alpha,
            pooled_lb_alpha_bad=pooled_lb_alpha_bad,
        )

    # ========== ivw_member_means (T2) ==========

    def _ivw_member_means_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        diagnostics: str = "compact",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute ivw_member_means (T2) statistics for one column."""
        return _ivw_member_means_stats(
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

    # Map canonical technique → metadata key (used by the per-stat helpers below).
    _METADATA_KEYS = {
        ENSEMBLE_AVERAGE: "technique_0_ensemble_average",
        POOLED_BLOCK_MEANS: "technique_1_pooled_block_means",
        IVW_MEMBER_MEANS: "technique_2_ivw_member_means",
    }

    def _per_stat_dispatch(
        self,
        cols,
        stat_key,
        default,
        ddof: int,
        method: str,
        window_size: Optional[int],
        technique,
        diagnostics: str,
    ) -> Dict:
        """Internal helper used by mean / mean_uncertainty / confidence_interval."""
        canonical = _normalize_technique(technique)
        result: Dict[str, Any] = {}
        meta_cols: Dict[str, Any] = {}

        if canonical == ENSEMBLE_AVERAGE:
            avg_ds = self.compute_average_ensemble()
            for col in cols:
                s, m = self._ensemble_average_stats_for_col(
                    col=col, ddof=ddof, method=method,
                    window_size=window_size, avg_ds=avg_ds,
                )
                result[col] = s.get(stat_key, default)
                meta_cols[col] = m
        elif canonical == POOLED_BLOCK_MEANS:
            for col in cols:
                s, m = self._pooled_block_means_stats_for_col(
                    col=col, ddof=ddof, window_size=window_size, method=method,
                )
                result[col] = s.get(stat_key, default)
                meta_cols[col] = m
        else:  # IVW_MEMBER_MEANS
            for col in cols:
                s, m = self._ivw_member_means_stats_for_col(
                    col=col, ddof=ddof, method=method,
                    window_size=window_size, diagnostics=diagnostics,
                )
                result[col] = s.get(stat_key, default)
                meta_cols[col] = (
                    m if diagnostics == "full"
                    else {"n_members_used": m.get("n_members_used", 0)}
                )

        return {
            "results": result,
            "metadata": {self._METADATA_KEYS[canonical]: meta_cols},
        }

    def mean(
        self,
        column_name=None,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique=POOLED_BLOCK_MEANS,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Ensemble mean.

        technique
            ``"ensemble_average"``     — build single averaged trace, compute stats on it
            ``"pooled_block_means"``   — preferred for trimmed ensembles
            ``"ivw_member_means"``     — member-wise then inverse-variance aggregate
            (legacy ``0/1/2`` and ``"technique0"/1/2`` strings still accepted)
        """
        return self._per_stat_dispatch(
            cols=self._resolve_cols(column_name),
            stat_key="mean",
            default=np.nan,
            ddof=1,
            method=method,
            window_size=window_size,
            technique=technique,
            diagnostics=diagnostics,
        )

    def mean_uncertainty(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique=POOLED_BLOCK_MEANS,
        diagnostics: str = "compact",
    ) -> Dict:
        """Ensemble SEM. See `mean()` for technique semantics."""
        return self._per_stat_dispatch(
            cols=self._resolve_cols(column_name),
            stat_key="mean_uncertainty",
            default=np.nan,
            ddof=ddof,
            method=method,
            window_size=window_size,
            technique=technique,
            diagnostics=diagnostics,
        )

    def confidence_interval(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique=POOLED_BLOCK_MEANS,
        diagnostics: str = "compact",
    ) -> Dict:
        """Ensemble confidence interval. See `mean()` for technique semantics."""
        return self._per_stat_dispatch(
            cols=self._resolve_cols(column_name),
            stat_key="confidence_interval",
            default=(np.nan, np.nan),
            ddof=ddof,
            method=method,
            window_size=window_size,
            technique=technique,
            diagnostics=diagnostics,
        )

    # ========== Full statistics ==========

    def compute_statistics(
        self,
        column_name=None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique=POOLED_BLOCK_MEANS,
        diagnostics: str = "compact",
        confidence_level: float = 0.95,
        ci_method: str = "normal",
    ) -> Dict:
        """
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
            confidence_level=confidence_level,
            ci_method=ci_method,
        )

    def compute_uncertainty(
        self,
        method="pooled_block_means",
        column_name=None,
        *,
        ddof: int = 1,
        window_size: Optional[int] = None,
        diagnostics: str = "compact",
        confidence_level: float = 0.95,
        ci_method: str = "normal",
    ) -> Dict:
        """Friendly alias for :meth:`compute_statistics` keyed by estimator name.

        ``method`` is the estimator: ``"ensemble_average"`` | ``"pooled_block_means"``
        | ``"ivw"`` (plus the legacy ``technique`` aliases). Equivalent to calling
        ``compute_statistics(..., technique=method)`` — the latter still works.

        Returns the same ``{"results": {...}, "metadata": {...}}`` schema.
        """
        return self.compute_statistics(
            column_name=column_name,
            ddof=ddof,
            window_size=window_size,
            technique=method,
            diagnostics=diagnostics,
            confidence_level=confidence_level,
            ci_method=ci_method,
        )

    # ========== ESS ==========

    def effective_sample_size(
        self,
        column_names=None,
        alpha: float = 0.05,
        technique=POOLED_BLOCK_MEANS,
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
        technique=POOLED_BLOCK_MEANS,
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
        technique=POOLED_BLOCK_MEANS,
    ) -> Dict:
        """Count of block means from compute_statistics."""
        out = self.compute_statistics(
            column_name=column_name, ddof=ddof, method=method,
            window_size=window_size, technique=technique,
        )
        res = {col: (s.get("n_short_averages", np.nan) if isinstance(s, dict) else np.nan)
               for col, s in out["results"].items()}
        return {"results": res, "metadata": out.get("metadata", {})}
