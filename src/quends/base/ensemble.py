# ============================================================
# ensemble.py
# ============================================================

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from quends.base.data_stream import DataStream


class Ensemble:
    """
    Manages an ensemble of (already-trimmed) DataStream instances and provides
    ensemble-level statistics.

    Key points for Technique 1 in mean/uncertainty/ci:
      - Assumes *already-trimmed* members.
      - Forms non-overlapping block means per member ONCE (window derived by each member's
        DataStream._estimate_window(...), unless a window_size is explicitly provided).
      - Concatenates those block means and computes:
          * mean = average(pooled_blocks)
          * SEM  = std(pooled_blocks, ddof)/sqrt(ESS_blocks)  (ESS via Geyer on pooled blocks)
          * CI   = mean ± 1.96*SEM
      - No re-processing after concatenation.

    For Technique 2 (member-wise then aggregate):
      - Uses each member's DataStream.compute_statistics(...)
      - Aggregates member results (simple averages).
      - Diagnostics behavior:
          * diagnostics='compact' -> do NOT include per-member histories
          * diagnostics='full'    -> include each member's _history
    """

    # ---------- Construction & basic access ----------
    def __init__(self, data_streams: List[DataStream]):
        if not isinstance(data_streams, list) or not data_streams:
            raise ValueError("Provide a non-empty list of DataStream objects.")
        if not all(isinstance(ds, DataStream) for ds in data_streams):
            raise ValueError(
                "All ensemble members must be DataStream instances."
            )
        self.data_streams = data_streams

    def __len__(self) -> int:
        return len(self.data_streams)

    def head(self, n: int = 5) -> Dict[int, pd.DataFrame]:
        return {i: ds.head(n) for i, ds in enumerate(self.data_streams)}

    def get_member(self, index: int) -> DataStream:
        return self.data_streams[index]

    def members(self) -> List[DataStream]:
        return self.data_streams

    # ---------- Variables & summary ----------
    def common_variables(self) -> List[str]:
        if not self.data_streams:
            return []
        sets = [set(ds.df.columns) - {"time"} for ds in self.data_streams]
        return sorted(list(set.intersection(*sets))) if sets else []

    def summary(self) -> Dict:
        info = {}
        for i, ds in enumerate(self.data_streams):
            info[f"Member {i}"] = {
                "n_samples": len(ds.df),
                "columns": list(ds.df.columns),
                "head": ds.head().to_dict(orient="list"),
            }
        out = {
            "n_members": len(self.data_streams),
            "common_variables": self.common_variables(),
            "members": info,
        }
        # Light printout (optional)
        print(f"Ensemble members: {len(self.data_streams)}")
        print("Common variables:", out["common_variables"])
        return out

    # ---------- Helpers ----------
    @staticmethod
    def _geyer_ess_on_blocks(block_means: np.ndarray) -> float:
        x = np.asarray(block_means, dtype=float)
        n = x.size
        if n <= 2:
            return 1.0
        r = acf(x, nlags=max(1, n // 4), fft=False)
        s, t = 0.0, 1
        while t + 1 < len(r):
            pair_sum = r[t] + r[t + 1]
            if pair_sum < 0:
                break
            s += pair_sum
            t += 2
        return max(1.0, n / (1.0 + 2.0 * s))

    def _resolve_cols(
        self, column_name: Optional[str | List[str]]
    ) -> List[str]:
        return (
            [column_name]
            if isinstance(column_name, str)
            else (
                self.common_variables()
                if column_name is None
                else list(column_name)
            )
        )

    def _autotune_member_blocks_until_independent(
        self,
        ds: DataStream,
        col: str,
        window_size: Optional[int],
        method: str = "non-overlapping",
        lb_alpha: float = 0.05,
        lb_lags: Optional[int] = None,
        max_tries: int = 25,
        grow_factor: float = 1.0,
        min_blocks: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        For ONE member: choose a window so that block means are independent
        (Ljung–Box passes) or return best_p if it never passes.

        Returns
        -------
        block_means : np.ndarray
        meta : Dict with keys:
            status: "independent" or "best_p"
            window_used, n_blocks, ljungbox_pvalue, ljungbox_lags, tries
            best_pvalue, best_window, warnings
        """
        out_meta: Dict[str, Any] = {
            "status": None,
            "window_used": None,
            "n_blocks": 0,
            "ljungbox_pvalue": np.nan,
            "ljungbox_lags": 0,
            "tries": 0,
            "best_pvalue": -np.inf,
            "best_window": None,
            "warnings": [],
        }

        if col not in ds.df.columns:
            out_meta["status"] = "best_p"
            out_meta["warnings"].append("column_missing")
            return np.array([], dtype=float), out_meta

        series = ds.df[col].dropna()
        n = int(series.size)
        if n < 4:
            out_meta["status"] = "best_p"
            out_meta["warnings"].append("too_few_samples")
            return np.array([], dtype=float), out_meta

        # starting window: user window if provided, else ds estimate
        w0 = ds._estimate_window(col, series, window_size)
        w = int(max(2, w0))

        best_blocks = np.array([], dtype=float)

        # helper: always increase w by at least 1
        def _next_w(cur: int) -> int:
            if grow_factor is None or grow_factor <= 1.0:
                return cur + 1
            nxt = int(np.ceil(cur * float(grow_factor)))
            return max(cur + 1, nxt)

        for t in range(max_tries):
            out_meta["tries"] = t + 1

            if w >= n:
                break
            bm = ds._process_column(series, estimated_window=w, method=method)
            if bm is None or len(bm) == 0:
                # try growing window anyway
                w = _next_w(w)  # w + grow_factor #int(np.ceil(w * grow_factor))
                continue

            blocks = np.asarray(bm.values, dtype=float)
            n_blocks = int(blocks.size)

            # If we can't even form enough blocks, LB is not meaningful.
            # Increasing w will only reduce blocks further → stop.
            if n_blocks < max(3, min_blocks):
                # growing window will only reduce blocks further, so we stop
                if best_blocks.size == 0:
                    best_blocks = blocks
                    out_meta["best_window"] = int(w)
                    out_meta["best_pvalue"] = np.nan
                out_meta["warnings"].append("too_few_blocks_for_ljungbox")
                break

            # choose lags (like your earlier pattern)
            lags = lb_lags
            if lags is None:
                lags = max(1, min(20, n_blocks // 4))

            try:
                lb = acorr_ljungbox(blocks, lags=[lags], return_df=True)
                pval = float(lb["lb_pvalue"].iloc[0])
            except Exception:
                pval = np.nan
            # track best p-value window regardless of pass/fail
            if np.isfinite(pval) and pval > float(out_meta["best_pvalue"]):
                out_meta["best_pvalue"] = pval
                out_meta["best_window"] = int(w)
                best_blocks = blocks

            # pass?
            if np.isfinite(pval) and pval >= lb_alpha:
                out_meta.update(
                    {
                        "status": "independent",
                        "window_used": int(w),
                        "n_blocks": int(n_blocks),
                        "ljungbox_pvalue": float(pval),
                        "ljungbox_lags": int(lags),
                    }
                )
                return blocks, out_meta
            # fail -> grow window and try again
            w = _next_w(w)
            # w = w + grow_factor #int(np.ceil(w * grow_factor))

        # if we get here: never passed, use best_p if we have it
        if best_blocks.size > 0 and out_meta["best_window"] is not None:
            # recompute LB diagnostics at best window (optional but nice)
            n_blocks = int(best_blocks.size)
            lags = (
                lb_lags
                if lb_lags is not None
                else max(1, min(20, n_blocks // 4))
            )
            try:
                lb = acorr_ljungbox(best_blocks, lags=[lags], return_df=True)
                pval = float(lb["lb_pvalue"].iloc[0])
            except Exception:
                pval = np.nan
            out_meta.update(
                {
                    "status": "best_p",
                    "window_used": int(out_meta["best_window"]),
                    "n_blocks": int(n_blocks),
                    "ljungbox_pvalue": float(pval),
                    "ljungbox_lags": int(lags),
                }
            )
            out_meta["warnings"].append(
                "independence_failed_used_best_p_window"
            )
            return best_blocks, out_meta

        out_meta["status"] = "best_p"
        out_meta["warnings"].append("no_valid_blocks")
        return np.array([], dtype=float), out_meta

    # ============================================================
    # 1) Technique 1: pooled-block statistics
    # ============================================================
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
        """
        Compute Technique-1 (pooled-block) statistics ONCE for a single column,
        mirroring DataStream.compute_statistics style.

        Policy:
        - Always compute BOTH sem_n and sem_ess.
        - Default reported SEM uses sem_n if:
            (a) ALL members are "independent", OR
            (b) SOME are "best_p" (warn).
        - Final pooled Ljung–Box diagnostic:
            If pooled p-value < pooled_lb_alpha_bad, switch reported SEM to sem_ess
            (ultra conservative), but still keep sem_n in output.

        Returns
        -------
        stats : Dict
            mean, variance, ess_blocks, mean_uncertainty (reported),
            mean_uncertainty_sem_n, mean_uncertainty_sem_ess,
            confidence_interval, pm_std, n_short_averages,
            pooled independence diagnostics, and se_method + warning.
        meta : Dict
            pooled-block metadata (member windows, blocks, member_status, etc.)
        # NOTE: method is intentionally ignored; Technique-1 is defined on non-overlapping blocks.
        """
        # Technique 1 should be based on non-overlapping blocks (true block means)

        pool_method = "non-overlapping"  # enforce true short-term averages

        pooled, meta = self._pooled_block_means_from_trimmed(
            col=col,
            window_size=window_size,
            method=pool_method,
            lb_alpha=lb_alpha,
            lb_lags=lb_lags,
        )

        out: Dict[str, Any] = {}
        # Pull member status flags (for warnings / reporting policy)
        # member_status examples: ["independent","independent","best_p",...]

        member_status = meta.get("member_status", [])
        all_independent = bool(member_status) and all(
            s == "independent" for s in member_status
        )
        some_best_p = any(s == "best_p" for s in member_status)

        if pooled.size == 0:
            out.update(
                dict(
                    mean=np.nan,
                    variance=np.nan,
                    ess_blocks=np.nan,
                    n_short_averages=0,
                    mean_uncertainty=np.nan,
                    mean_uncertainty_sem_n=np.nan,
                    mean_uncertainty_sem_ess=np.nan,
                    se_method=None,
                    warning="No pooled block means available.",
                    confidence_interval=(np.nan, np.nan),
                    pm_std=(np.nan, np.nan),
                    independent=None,
                    ljungbox_pvalue=np.nan,
                    ljungbox_lags=0,
                    member_all_independent=all_independent,
                    member_some_best_p=some_best_p,
                )
            )
            meta["pooled_blocks"] = 0
            return out, meta

        n_blocks = int(pooled.size)
        mu = float(np.mean(pooled))
        var_blocks = float(np.var(pooled, ddof=ddof)) if n_blocks > 1 else 0.0

        ess_blocks = float(max(1.0, float(self._geyer_ess_on_blocks(pooled))))
        sem_n = float(np.sqrt(var_blocks) / np.sqrt(max(1.0, float(n_blocks))))
        sem_ess = float(np.sqrt(var_blocks) / np.sqrt(ess_blocks))

        if lb_lags is None:
            lb_lags = int(max(1, min(20, n_blocks // 4)))

        try:
            lb = acorr_ljungbox(pooled, lags=[lb_lags], return_df=True)
            pooled_pval = float(lb["lb_pvalue"].iloc[0])
            pooled_independent = bool(pooled_pval >= lb_alpha)
        except Exception:
            pooled_pval = np.nan
            pooled_independent = None

        sem_reported = sem_n
        se_method = "sem_n"
        warning = None

        if some_best_p:
            warning = (
                "Some members did not pass Ljung–Box. Using their 'best_p' window block means. "
                "Reporting SEM via sem_n = sd(blocks)/sqrt(n_blocks). "
                "Also reporting sem_ess = sd(blocks)/sqrt(ESS_blocks) for conservative fallback."
            )

        # Final pooled check: if it fails *badly*, switch to sem_ess (ultra conservative)
        if np.isfinite(pooled_pval) and pooled_pval < float(
            pooled_lb_alpha_bad
        ):
            sem_reported = sem_ess
            se_method = "sem_ess (pooled_LB_bad)"
            extra = (
                f"Pooled Ljung–Box p-value={pooled_pval:.3g} < {pooled_lb_alpha_bad:.3g}; "
                "switching reported SEM to ESS-based (ultra conservative)."
            )
            warning = extra if warning is None else (warning + " " + extra)

        ci = (float(mu - 1.96 * sem_reported), float(mu + 1.96 * sem_reported))
        pm_std = (float(mu - sem_reported), float(mu + sem_reported))

        out.update(
            dict(
                mean=mu,
                variance=var_blocks,  # Var(pooled block means)
                ess_blocks=ess_blocks,  # diagnostic + conservative fallback
                n_short_averages=n_blocks,  # number of pooled block means
                # SEMs: report both + indicate which one is used
                mean_uncertainty=sem_reported,  # reported SEM (policy-based)
                mean_uncertainty_sem_n=sem_n,  # sd(blocks)/sqrt(n_blocks
                mean_uncertainty_sem_ess=sem_ess,  # sd(blocks)/sqrt(ESS_blocks)
                se_method=se_method,
                warning=warning,
                confidence_interval=ci,
                pm_std=pm_std,
                # pooled independence diagnostic (optional)
                independent=pooled_independent,
                ljungbox_pvalue=pooled_pval,
                ljungbox_lags=int(lb_lags),
                # member status summary
                member_all_independent=all_independent,
                member_some_best_p=some_best_p,
            )
        )

        meta["pooled_blocks"] = n_blocks
        return out, meta

    # ============================================================
    # 2) FIX: pooled blocks builder must call self._autotune...
    # ============================================================
    def _pooled_block_means_from_trimmed(
        self,
        col: str,
        window_size: Optional[int],
        method: str = "non-overlapping",
        lb_alpha: float = 0.05,
        lb_lags: Optional[int] = None,
        max_tries: int = 25,
        grow_factor: float = 1.0,
        min_blocks: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        For ALL members: get pooled block means for a given column,
        autotuning each member's window until block means are independent.

        Returns
        -------
        pooled_blocks : np.ndarray
            Concatenated block means from all members (float dtype).
        meta : Dict with keys:
            members_used, member_windows, member_blocks, member_status,
            member_pvalues, member_lags, member_warnings
        NEW behavior (Technique-1 pooling, DataStream-aligned):
        - For each member, autotune window until block means pass Ljung–Box
            ("independent") or return "best_p" (best achievable p-value).
        - Pool ONLY those per-member block means (short-term averages).
        - Return pooled blocks + rich per-member metadata.

        Notes:
        - We intentionally use non-overlapping blocks for "short-term averages".
        - Independence is enforced PER MEMBER before pooling.
        - Pooled independence can still be tested later (diagnostic).
        """

        pooled: List[np.ndarray] = []
        meta: Dict[str, Any] = dict(
            members_used=0,
            member_windows=[],
            member_blocks=[],
            member_status=[],
            member_pvalues=[],
            member_lags=[],
            member_warnings=[],
        )

        pool_method = "non-overlapping" if method is None else method

        for j, ds in enumerate(self.data_streams):
            blocks, m = self._autotune_member_blocks_until_independent(
                ds=ds,
                col=col,
                window_size=window_size,
                method=pool_method,
                lb_alpha=lb_alpha,
                lb_lags=lb_lags,
                max_tries=max_tries,
                grow_factor=grow_factor,
                min_blocks=min_blocks,
            )
            if blocks is None or blocks.size == 0:
                continue

            pooled.append(blocks.astype(float, copy=False))

            # ---- member metadata ----
            status = m.get("status") or "unknown"
            warnings = m.get("warnings", [])
            if warnings is None:
                warnings = []
            elif not isinstance(warnings, list):
                warnings = [str(warnings)]

            meta["members_used"] += 1
            meta["member_windows"].append(int(m.get("window_used", -1) or -1))
            meta["member_blocks"].append(int(m.get("n_blocks", 0) or 0))
            meta["member_status"].append(status)
            meta["member_pvalues"].append(
                float(m.get("ljungbox_pvalue", np.nan))
            )
            meta["member_lags"].append(int(m.get("ljungbox_lags", 0) or 0))
            meta["member_warnings"].append(warnings)

        if not pooled:
            return np.array([], dtype=float), meta

        return np.concatenate(pooled, axis=0), meta

    # ============================================================
    # 3) NEW: Technique-2 (member-wise, simple averages; NOT pooled)
    # ============================================================
    def _tech2_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        diagnostics: str = "compact",  # "compact" | "full"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Technique 2 (member-wise then aggregate) computed ONCE for a single column.

        What it does (current aggregation rule):
        - For each member: call DataStream.compute_statistics(...) for `col`
        - Aggregate ensemble outputs by *simple averages* across members:
            mean = avg(member means)
            mean_uncertainty = avg(member SEs)
            CI = avg(lower bounds), avg(upper bounds)

        Aggregation rule (means across members):
            - mean                = mean(member means)
            - mean_uncertainty    = mean(member mean_uncertainty)
            - confidence_interval = (mean(CI_low_i), mean(CI_high_i))
            - ess_blocks          = mean(member ess_blocks)           (new)
            - n_short_averages    = mean(member n_short_averages)     (new)
            - variance            = mean(member variances)

        NOTE:
        - DataStream variance here is Var(block_means), not raw-series variance.
        - This keeps Technique-2 intentionally simple (not pooled).
        If later you want a more “statistically pooled” version of Technique-2,
        we'd change aggregation rules (e.g., pooling block means or using inverse-variance weights).

        Returns
        -------
        stats : Dict with keys:
            mean, mean_uncertainty, confidence_interval, pm_std,
            ensemble_ess, individual (per-member stats)
        meta : Dict with keys:
            ess_blocks, n_short_averages, individual (per-member stats)
        """
        indiv: Dict[str, Any] = {}
        histories: Dict[str, Any] = {}

        means: List[float] = []
        ses: List[float] = []
        ci_lows: List[float] = []
        ci_highs: List[float] = []

        var_list: List[float] = []
        ess_list: List[float] = []
        nsa_list: List[float] = []

        for i, ds in enumerate(self.data_streams):
            key = f"Member {i}"
            if col not in ds.df.columns:
                continue

            # IMPORTANT: call once per member for this col
            s = ds.compute_statistics(
                column_name=col,
                ddof=ddof,
                method=method,
                window_size=window_size,
                diagnostics="none",
            )
            # DataStream.compute_statistics returns {col: {...}}
            stat = s.get(col, {})
            if not isinstance(stat, dict):
                continue

            mu = stat.get("mean", np.nan)
            se = stat.get("mean_uncertainty", np.nan)
            ci = stat.get("confidence_interval", (np.nan, np.nan))
            pm = stat.get("pm_std", (np.nan, np.nan))

            # Accept either key just in case
            ess_i = stat.get(
                "ess_blocks", stat.get("block_effective_n", np.nan)
            )
            nsa_i = stat.get("n_short_averages", np.nan)

            # NEW: pull member variance of block means
            var_i = stat.get("variance", np.nan)

            indiv[key] = {
                "mean": mu,
                "mean_uncertainty": se,
                "variance": var_i,
                "confidence_interval": ci,
                "pm_std": pm,
                "ess_blocks": ess_i,
                "n_short_averages": nsa_i,
            }
            if np.isfinite(mu):
                means.append(float(mu))
            if np.isfinite(se):
                ses.append(float(se))
            if np.isfinite(var_i):
                var_list.append(float(var_i))
            if np.isfinite(ess_i):
                ess_list.append(float(ess_i))
            if np.isfinite(nsa_i):
                nsa_list.append(float(nsa_i))

            if isinstance(ci, (tuple, list)) and len(ci) == 2:
                lo, hi = ci
                if np.isfinite(lo):
                    ci_lows.append(float(lo))
                if np.isfinite(hi):
                    ci_highs.append(float(hi))

            if diagnostics == "full":
                histories[key] = getattr(ds, "_history", None)

        # simple aggregation (old behavior)
        mu_ens = float(np.nanmean(means)) if means else np.nan
        se_ens = float(np.nanmean(ses)) if ses else np.nan
        var_ens = float(np.nanmean(var_list)) if var_list else np.nan
        ci_ens = (
            float(np.nanmean(ci_lows)) if ci_lows else np.nan,
            float(np.nanmean(ci_highs)) if ci_highs else np.nan,
        )

        # NEW: mean ESS + mean n_short_averages across members
        ess_ens = float(np.nanmean(ess_list)) if ess_list else np.nan
        nsa_ens = float(np.nanmean(nsa_list)) if nsa_list else np.nan

        out: Dict[str, Any] = {
            "mean": mu_ens,
            "mean_uncertainty": se_ens,
            "confidence_interval": ci_ens,
            "pm_std": (
                (
                    (mu_ens - se_ens)
                    if np.isfinite(mu_ens) and np.isfinite(se_ens)
                    else np.nan
                ),
                (
                    (mu_ens + se_ens)
                    if np.isfinite(mu_ens) and np.isfinite(se_ens)
                    else np.nan
                ),
            ),
            # Keep schema compatible with Technique-1 outputs:
            "variance": var_ens,
            "ess_blocks": ess_ens,
            "n_short_averages": nsa_ens,
            "se_method": "tech2_simple_member_average",
            "warning": None,
            "individual": (indiv if diagnostics == "full" else None),
        }

        meta: Dict[str, Any] = {"n_members_used": len(indiv)}
        if diagnostics == "full":
            meta["individual_histories"] = histories

        return out, meta

    # ---------- Average-ensemble utilities (kept; independent of techniques) ----------
    def compute_average_ensemble(
        self,
        members: Optional[List[DataStream]] = None,
        interp_method: str = "spline",
        verbose: bool = False,
    ) -> DataStream:
        """
        Interpolate each member to a common regular grid and average common variables.
        """
        data_streams = members if members is not None else self.data_streams
        if not data_streams:
            raise ValueError("No data streams provided for ensemble averaging.")

        interpolated_ensemble, interp_meta = self.__class__(
            data_streams
        ).interpolate_to_common_time(
            column_name="time", method=interp_method, verbose=verbose
        )
        dfs = [ds.df for ds in interpolated_ensemble.data_streams]
        common_times = dfs[0]["time"].values

        avg_df = pd.DataFrame({"time": common_times})
        common_cols = interpolated_ensemble.common_variables()
        for col in common_cols:
            arrays = np.stack([df[col].values for df in dfs])
            avg_df[col] = np.nanmean(arrays, axis=0)

        mask = ~avg_df[common_cols].isna().all(axis=1)
        avg_df = avg_df.loc[mask].reset_index(drop=True)

        history = []
        for ds in interpolated_ensemble.data_streams:
            if hasattr(ds, "_history"):
                history.extend(ds._history)
        history.append(
            {
                "operation": "ensemble_average_on_interpolated_grid",
                "options": {
                    "n_members": len(dfs),
                    "common_columns": common_cols,
                    "interp_method": interp_method,
                },
                "interpolation_metadata": interp_meta,
            }
        )
        return DataStream(avg_df, _history=history)

    def compute_average_ensemble_unaligned(
        self,
        column_name: Optional[str | List[str]] = None,
        min_coverage: int = 1,
        verbose: bool = False,
    ) -> Tuple[DataStream, pd.Series]:
        """
        Average across members without interpolation (unaligned time axes).
        """
        if column_name is None:
            cols = self.common_variables()
        elif isinstance(column_name, str):
            cols = [column_name]
        else:
            cols = column_name

        # Union of rounded times for robustness
        all_times = np.unique(
            np.concatenate(
                [np.round(ds.df["time"].values, 6) for ds in self.data_streams]
            )
        )
        all_times.sort()
        avg_df = pd.DataFrame({"time": all_times})

        for col in cols:
            member_cols = []
            for ds in self.data_streams:
                if col not in ds.df.columns:
                    continue
                sub_df = ds.df[["time", col]].dropna().copy()
                sub_df["time"] = np.round(sub_df["time"], 6)
                series = pd.Series(index=all_times, dtype=float)
                for t, v in zip(sub_df["time"].values, sub_df[col].values):
                    idxs = np.where(np.isclose(all_times, t, atol=1e-5))[0]
                    if len(idxs) > 0:
                        series.iloc[idxs[0]] = v
                member_cols.append(series)

            if len(member_cols) == 0:
                avg_df[col] = np.nan
                continue

            stack = pd.concat(member_cols, axis=1)
            mean_vals = stack.mean(axis=1, skipna=True)
            count_vals = stack.count(axis=1)
            avg_df[col] = mean_vals.where(count_vals >= min_coverage, np.nan)
            coverage = count_vals  # last computed; ok for quick inspection

            if verbose:
                print(
                    f"[{col}] coverage: min={int(count_vals.min())}, max={int(count_vals.max())}"
                )

        avg_df = avg_df.dropna(subset=cols, how="all").reset_index(drop=True)
        return DataStream(avg_df), (
            coverage if "coverage" in locals() else pd.Series(dtype=int)
        )

    # ---------- Utility ----------
    @staticmethod
    def collect_histories(ds_list: List[DataStream]) -> List[List[dict]]:
        return [getattr(ds, "_history", []) for ds in ds_list]

    # ---------- Trim pass-through ----------
    def trim(
        self,
        column_name: str,
        batch_size: int = 10,
        start_time: float = 0.0,
        method: str = "std",
        threshold: Optional[float] = None,
        robust: bool = True,
    ) -> "Ensemble":
        """
        Call DataStream.trim(...) on each member and keep only non-empty results.
        """
        trimmed = [
            ds.trim(
                column_name=column_name,
                batch_size=batch_size,
                start_time=start_time,
                method=method,
                threshold=threshold,
                robust=robust,
            )
            for ds in self.data_streams
        ]
        kept = [
            t
            for t in trimmed
            if t is not None and hasattr(t, "df") and not t.df.empty
        ]
        if not kept:
            raise ValueError(
                "No ensemble members survived trimming (all failed or empty)!"
            )
        return Ensemble(kept)

    # ---------- Stationarity ----------
    def is_stationary(self, columns) -> Dict:
        results, meta = {}, {}
        for i, ds in enumerate(self.data_streams):
            r = ds.is_stationary(columns)
            results[f"Member {i}"] = r
            meta[f"Member {i}"] = getattr(ds, "_history", None)
        return {"results": results, "metadata": meta}

    # ---------- MEAN / SEM / CI ----------
    def mean(
        self,
        column_name: Optional[str | List[str]] = None,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Ensemble mean.

        Techniques:
          1 - pooled block-means from already-trimmed members (preferred)
          2 - member-wise mean then aggregate

        diagnostics: 'none' | 'compact' | 'full'
        """
        metadata: Dict = {}

        cols = self._resolve_cols(column_name)
        # cols = (
        #     [column_name]
        #     if isinstance(column_name, str)
        #     else (
        #         self.common_variables()
        #         if column_name is None
        #         else list(column_name)
        #     )
        # )

        if technique == 1:
            result = {}
            meta_cols = {}
            for col in cols:
                s, m = self._tech1_pooled_stats_for_col(
                    col=col, ddof=1, window_size=window_size, method=method
                )
                result[col] = s.get("mean", np.nan)
                meta_cols[col] = m
            metadata["technique_1_pooled"] = meta_cols
            return {"results": result, "metadata": metadata}

        elif technique == 2:
            result = {}
            meta_cols = {}
            for col in cols:
                s, m = self._tech2_stats_for_col(
                    col=col,
                    ddof=1,
                    method=method,
                    window_size=window_size,
                    diagnostics=diagnostics,
                )
                result[col] = s.get("mean", np.nan)
                meta_cols[col] = (
                    m
                    if diagnostics == "full"
                    else {"n_members_used": m.get("n_members_used", 0)}
                )
            metadata["technique_2_memberwise"] = meta_cols
            return {"results": result, "metadata": metadata}

        else:
            raise ValueError("Invalid technique. Use 1 or 2.")

    # ============================================================
    # mean_uncertainty() and confidence_interval()
    # (Technique 1 uses _tech1_pooled_stats_for_col)
    # (Technique 2 uses _tech2_memberwise_stats_for_col)
    # ============================================================
    def mean_uncertainty(
        self,
        column_name: Optional[str | List[str]] = None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Ensemble SEM (mean uncertainty).

        technique=1: uses pooled-block stats helper (compute once per col)
        technique=2: uses member-wise stats helper (compute once per col)

        Returns
        -------
        {"results": {col: sem, ...}, "metadata": {...}}
        """
        metadata: Dict = {}

        cols = self._resolve_cols(column_name)
        # cols = (
        #     [column_name]
        #     if isinstance(column_name, str)
        #     else (
        #         self.common_variables()
        #         if column_name is None
        #         else list(column_name)
        #     )
        # )

        result: Dict = {}
        meta_cols: Dict = {}

        if technique == 1:
            for col in cols:
                s, m = self._tech1_pooled_stats_for_col(
                    col=col,
                    ddof=ddof,
                    window_size=window_size,
                    method=method,  # pooled helper can ignore this and force non-overlap
                )
                result[col] = s.get("mean_uncertainty", np.nan)
                meta_cols[col] = m

            metadata["technique_1_pooled"] = meta_cols
            return {"results": result, "metadata": metadata}

        elif technique == 2:
            for col in cols:
                s, m = self._tech2_stats_for_col(
                    col=col,
                    ddof=ddof,
                    method=method,
                    window_size=window_size,
                    diagnostics=diagnostics,
                )
                result[col] = s.get("mean_uncertainty", np.nan)
                # keep metadata light unless full diagnostics requested
                meta_cols[col] = (
                    m
                    if diagnostics == "full"
                    else {"n_members_used": m.get("n_members_used", 0)}
                )

            metadata["technique_2_memberwise"] = meta_cols
            return {"results": result, "metadata": metadata}

        else:
            raise ValueError("Invalid technique. Use 1 or 2.")

    def confidence_interval(
        self,
        column_name: Optional[str | List[str]] = None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Ensemble 95% CI.

        technique=1: uses pooled-block stats helper (compute once per col)
        technique=2: uses member-wise stats helper (compute once per col)

        Returns
        -------
        {"results": {col: (ci_low, ci_high), ...}, "metadata": {...}}
        """
        metadata: Dict = {}

        cols = self._resolve_cols(column_name)
        # cols = (
        #     [column_name]
        #     if isinstance(column_name, str)
        #     else (
        #         self.common_variables()
        #         if column_name is None
        #         else list(column_name)
        #     )
        # )

        result: Dict = {}
        meta_cols: Dict = {}

        if technique == 1:
            for col in cols:
                s, m = self._tech1_pooled_stats_for_col(
                    col=col,
                    ddof=ddof,
                    window_size=window_size,
                    method=method,  # pooled helper can ignore this and force non-overlap
                )
                result[col] = s.get("confidence_interval", (np.nan, np.nan))
                meta_cols[col] = m

            metadata["technique_1_pooled"] = meta_cols
            return {"results": result, "metadata": metadata}

        elif technique == 2:
            for col in cols:
                s, m = self._tech2_stats_for_col(
                    col=col,
                    ddof=ddof,
                    method=method,
                    window_size=window_size,
                    diagnostics=diagnostics,
                )
                result[col] = s.get("confidence_interval", (np.nan, np.nan))
                meta_cols[col] = (
                    m
                    if diagnostics == "full"
                    else {"n_members_used": m.get("n_members_used", 0)}
                )

            metadata["technique_2_memberwise"] = meta_cols
            return {"results": result, "metadata": metadata}

        else:
            raise ValueError("Invalid technique. Use 1 or 2.")

    def compute_statistics(
        self,
        column_name: Optional[str | List[str]] = None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:

        cols = self._resolve_cols(column_name)
        # cols = (
        #     [column_name]
        #     if isinstance(column_name, str)
        #     else (
        #         self.common_variables()
        #         if column_name is None
        #         else list(column_name)
        #     )
        # )

        stats: Dict = {}
        metadata: Dict = {}

        if technique == 1:
            meta_cols = {}
            for col in cols:
                col_stats, col_meta = self._tech1_pooled_stats_for_col(
                    col=col,
                    ddof=ddof,
                    window_size=window_size,
                    method=method,  # (ignored inside if you force non-overlap)
                )
                stats[col] = col_stats
                meta_cols[col] = col_meta
            metadata["technique_1_pooled"] = meta_cols
            return {"results": stats, "metadata": metadata}

        elif technique == 2:
            meta_cols = {}
            for col in cols:
                col_stats, col_meta = self._tech2_stats_for_col(
                    col=col,
                    ddof=ddof,
                    method=method,
                    window_size=window_size,
                    diagnostics=diagnostics,
                )
                # Important: make it match your Technique-1 schema at top level
                # so downstream code can treat both techniques the same.
                stats[col] = {
                    "mean": col_stats["mean"],
                    "mean_uncertainty": col_stats["mean_uncertainty"],
                    "confidence_interval": col_stats["confidence_interval"],
                    "pm_std": col_stats["pm_std"],
                    "variance": col_stats.get("variance", np.nan),
                    "ess_blocks": col_stats.get("ess_blocks", np.nan),
                    "n_short_averages": col_stats.get(
                        "n_short_averages", np.nan
                    ),
                    "se_method": col_stats.get("se_method"),
                    "warning": col_stats.get("warning"),
                    # optionally include per-member breakdown
                    "individual": (
                        col_stats.get("individual", {})
                        if diagnostics == "full"
                        else None
                    ),
                }
                meta_cols[col] = col_meta
            metadata["technique_2_memberwise"] = meta_cols
            return {"results": stats, "metadata": metadata}

        else:
            raise ValueError("Invalid technique. Use 1 or 2.")

    # ---------- Interpolate all members to a common, regular time grid ----------
    def interpolate_to_common_time(
        self,
        column_name: str = "time",
        tol: float = 1e-8,
        method: str = "spline",  # 'linear' or 'spline'
        verbose: bool = True,
    ) -> Tuple["Ensemble", Dict]:
        """
        Interpolate all ensemble members onto a common, regular time grid.
        """
        step_info = []
        min_time, max_time = np.inf, -np.inf

        # 1) gather step diagnostics and time ranges
        for i, ds in enumerate(self.data_streams):
            checked = ds.check_time_steps_uniformity(
                column_name=column_name, tol=tol, print_details=False
            )
            step_info.append(getattr(checked, "_uniformity_result", checked))
            t0, t1 = ds.df[column_name].iloc[0], ds.df[column_name].iloc[-1]
            min_time = min(min_time, t0)
            max_time = max(max_time, t1)

        # 2) majority step across members
        all_steps = []
        for info in step_info:
            all_steps.extend(np.round(info["unique_steps"], 10))
        if not all_steps:
            raise RuntimeError(
                "No valid step information found for any ensemble member."
            )
        unique, counts = np.unique(all_steps, return_counts=True)
        majority_step = unique[np.argmax(counts)]

        # 3) build grid
        n_grid = int(np.ceil((max_time - min_time) / majority_step)) + 1
        common_times = min_time + np.arange(n_grid) * majority_step

        # 4) interpolate each member
        new_members: List[DataStream] = []
        interp_meta = []
        for idx, ds in enumerate(self.data_streams):
            orig_times = ds.df[column_name].values
            new_df = {column_name: common_times}
            for col in ds.df.columns:
                if col == column_name:
                    continue
                y = ds.df[col].values
                mask = np.isfinite(orig_times) & np.isfinite(y)
                if np.sum(mask) < 2:
                    new_df[col] = np.full_like(
                        common_times, np.nan, dtype=np.float64
                    )
                    continue
                try:
                    if method == "spline" and np.sum(mask) >= 3:
                        f = CubicSpline(
                            orig_times[mask], y[mask], extrapolate=True
                        )
                    else:
                        f = interp1d(
                            orig_times[mask],
                            y[mask],
                            kind="linear",
                            fill_value="extrapolate",
                        )
                    new_df[col] = f(common_times)
                except Exception as e:
                    if verbose:
                        print(
                            f"Interpolation failed for member {idx}, col {col}: {e}"
                        )
                    new_df[col] = np.full_like(
                        common_times, np.nan, dtype=np.float64
                    )

            new_history = deepcopy(ds._history)
            new_history.append(
                {
                    "operation": "interpolated_to_common_time_grid",
                    "options": {
                        "column_name": column_name,
                        "majority_step": float(majority_step),
                        "common_times": (float(min_time), float(max_time)),
                        "grid_len": int(len(common_times)),
                        "method": method,
                    },
                }
            )
            new_ds = ds.__class__(pd.DataFrame(new_df), _history=new_history)
            new_members.append(new_ds)
            interp_meta.append(
                {
                    "member": idx,
                    "step_status": step_info[idx].get("status"),
                    "original_steps": step_info[idx].get("unique_steps", []),
                }
            )

        diagnostics = {
            "majority_step": float(majority_step),
            "common_time_grid": common_times,
            "min_time": float(min_time),
            "max_time": float(max_time),
            "step_info": step_info,
            "interpolation_metadata": interp_meta,
        }

        if verbose:
            print(
                f"Built common time grid: min={min_time:.4g}, max={max_time:.4g}, "
                f"step={majority_step:.4g}, N={len(common_times)}"
            )
            print(f"Interpolated {len(new_members)} members.")

        return self.__class__(new_members), diagnostics

    # ============================================================
    # 5) NEW: convenience accessors
    #    - effective_sample_size_blocks(): returns ess_blocks from compute_statistics
    #    - n_short_averages(): returns n_short_averages (n_blocks) from compute_statistics
    # ============================================================
    def effective_sample_size_blocks(
        self,
        column_name: Optional[str | List[str]] = None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict[str, Any]:
        out = self.compute_statistics(
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
            technique=technique,
            diagnostics=diagnostics,
        )
        res = {}
        for col, s in out["results"].items():
            res[col] = (
                s.get("ess_blocks", np.nan) if isinstance(s, dict) else np.nan
            )
        return {"results": res, "metadata": out.get("metadata", {})}

    def n_short_averages(
        self,
        column_name: Optional[str | List[str]] = None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict[str, Any]:
        out = self.compute_statistics(
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
            technique=technique,
            diagnostics=diagnostics,
        )
        res = {}
        for col, s in out["results"].items():
            if not isinstance(s, dict):
                res[col] = np.nan
                continue
            # Technique 1: will exist as pooled n_short_averages
            if "n_short_averages" in s:
                res[col] = s.get("n_short_averages", np.nan)
            else:
                # Technique 2: optional (if you choose to store a count)
                res[col] = s.get("n_short_averages", np.nan)
        return {"results": res, "metadata": out.get("metadata", {})}
