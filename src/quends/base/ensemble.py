from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from statsmodels.stats.diagnostic import acorr_ljungbox

from quends.base.data_stream import DataStream
from quends.base.utils import _geyer_ess_on_blocks

"""
Module: ensemble.py

Three analysis techniques are available (passed as `technique` parameter):

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
"""


class Ensemble:
    """
    Manages an ensemble of DataStream instances for multi-stream analysis.
    """

    def __init__(self, data_streams: List[DataStream]):
        if not isinstance(data_streams, list) or not data_streams:
            raise ValueError("Provide a non-empty list of DataStream objects.")
        if not all(isinstance(ds, DataStream) for ds in data_streams):
            raise ValueError("All ensemble members must be DataStream instances.")
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
        if not self.data_streams:
            return []
        sets = [set(ds.data.columns) - {"time"} for ds in self.data_streams]
        return sorted(list(set.intersection(*sets))) if sets else []

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
        if isinstance(column_name, str):
            return [column_name]
        if column_name is None:
            return self.common_variables()
        return list(column_name)

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
        """
        member_info: Dict[str, Any] = {}
        all_steps: List[float] = []

        for i, ds in enumerate(self.data_streams):
            if "time" not in ds.data.columns:
                member_info[f"Member {i}"] = {"status": "no_time_column"}
                continue

            times = ds.data["time"].values
            if len(times) < 2:
                member_info[f"Member {i}"] = {"status": "too_few_points"}
                continue

            steps = np.diff(times)
            rounded = np.round(steps / tol) * tol
            uniq = np.unique(rounded).tolist()

            if len(uniq) == 1:
                status = "AllEqual"
            elif (
                len(uniq) == 2
                and np.allclose(rounded[:-1], rounded[0], atol=tol)
                and not np.isclose(rounded[-1], rounded[0], atol=tol)
            ):
                status = "AllEqualButLast"
            else:
                status = "NotUniform"

            all_steps.extend(uniq)
            member_info[f"Member {i}"] = {
                "status": status,
                "unique_steps": uniq,
                "n_steps": len(steps),
                "t_min": float(times[0]),
                "t_max": float(times[-1]),
            }

            if verbose:
                print(f"  Member {i}: status={status}, unique_steps={uniq}")

        # Majority step across all members
        if all_steps:
            step_arr = np.array(all_steps)
            vals, counts = np.unique(np.round(step_arr, 10), return_counts=True)
            majority_step = float(vals[np.argmax(counts)])
        else:
            majority_step = np.nan

        # Overall uniform: all members AllEqual with the same step
        all_equal = all(
            v.get("status") in ("AllEqual", "AllEqualButLast")
            and len(v.get("unique_steps", [])) == 1
            and np.isclose(v["unique_steps"][0], majority_step, atol=tol)
            for v in member_info.values()
            if "status" in v
        )

        return {
            "uniform": all_equal,
            "majority_step": majority_step,
            "members": member_info,
        }

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
        """
        step_info = self.check_time_steps_uniformity(tol=tol, verbose=verbose)
        majority_step = step_info["majority_step"]

        if not np.isfinite(majority_step) or majority_step <= 0:
            raise ValueError("Could not determine a valid majority step size.")

        # Time range across all members
        t_min = min(
            v["t_min"]
            for v in step_info["members"].values()
            if "t_min" in v
        )
        t_max = max(
            v["t_max"]
            for v in step_info["members"].values()
            if "t_max" in v
        )

        n_grid = int(np.ceil((t_max - t_min) / majority_step)) + 1
        common_times = t_min + np.arange(n_grid) * majority_step

        if verbose:
            print(
                f"Common grid: t_min={t_min:.4g}, t_max={t_max:.4g}, "
                f"step={majority_step:.4g}, N={n_grid}"
            )

        new_members: List[DataStream] = []
        interp_meta: List[Dict] = []

        for idx, ds in enumerate(self.data_streams):
            orig_times = ds.data["time"].values
            new_df: Dict[str, np.ndarray] = {"time": common_times}

            for col in ds.data.columns:
                if col == "time":
                    continue
                y = ds.data[col].values
                mask = np.isfinite(orig_times) & np.isfinite(y)
                if np.sum(mask) < 2:
                    new_df[col] = np.full(len(common_times), np.nan)
                    continue
                try:
                    if method == "spline" and np.sum(mask) >= 4:
                        f = CubicSpline(orig_times[mask], y[mask], extrapolate=True)
                    else:
                        f = interp1d(
                            orig_times[mask], y[mask],
                            kind="linear", fill_value="extrapolate",
                            bounds_error=False,
                        )
                    new_df[col] = f(common_times)
                except Exception as e:
                    if verbose:
                        print(f"  Interpolation failed for member {idx}, col {col}: {e}")
                    new_df[col] = np.full(len(common_times), np.nan)

            new_ds = DataStream(pd.DataFrame(new_df))
            new_members.append(new_ds)
            interp_meta.append({
                "member": idx,
                "step_status": step_info["members"].get(f"Member {idx}", {}).get("status"),
                "original_t_range": (
                    float(orig_times[0]) if len(orig_times) else np.nan,
                    float(orig_times[-1]) if len(orig_times) else np.nan,
                ),
            })

        diagnostics = {
            "majority_step": majority_step,
            "t_min": float(t_min),
            "t_max": float(t_max),
            "n_grid": int(n_grid),
            "method": method,
            "step_info": step_info,
            "member_meta": interp_meta,
        }
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
        """
        data_streams = members if members is not None else self.data_streams
        if not data_streams:
            raise ValueError("No data streams provided for ensemble averaging.")

        # Check if grids are already aligned
        sub_ens = self.__class__(data_streams)
        step_info = sub_ens.check_time_steps_uniformity(tol=tol)

        if step_info["uniform"]:
            # All members share the same regular grid — average directly
            avg_streams, _ = self._direct_average(data_streams, min_coverage=min_coverage)
            return avg_streams

        # Grids differ — interpolate first
        if verbose:
            print("Time grids differ; interpolating to common grid.")
        interp_ens, diag = sub_ens.interpolate_to_common_time(
            method=interp_method, tol=tol, verbose=verbose
        )
        avg_streams, _ = self._direct_average(interp_ens.data_streams, min_coverage=min_coverage)
        return avg_streams

    def _direct_average(
        self,
        data_streams: List[DataStream],
        min_coverage: int = 1,
    ) -> Tuple[DataStream, Dict]:
        """
        Average a list of DataStreams with compatible time grids by stacking
        and computing the per-time-point mean.

        Returns (averaged_DataStream, meta).
        """
        cols = self._resolve_cols(None)
        all_times = np.unique(
            np.concatenate(
                [np.round(ds.data["time"].values, 6) for ds in data_streams]
            )
        )
        all_times.sort()
        avg_df = pd.DataFrame({"time": all_times})

        for col in cols:
            member_series = []
            for ds in data_streams:
                if col not in ds.data.columns:
                    continue
                sub = ds.data[["time", col]].dropna().copy()
                sub["time"] = np.round(sub["time"], 6)
                s = pd.Series(np.nan, index=np.arange(len(all_times)), dtype=float)
                t_vals = sub["time"].to_numpy(dtype=float)
                v_vals = sub[col].to_numpy(dtype=float)
                idxs = np.searchsorted(all_times, t_vals)
                valid = (idxs >= 0) & (idxs < len(all_times))
                if np.any(valid):
                    s.iloc[idxs[valid]] = v_vals[valid]
                member_series.append(s)

            if not member_series:
                avg_df[col] = np.nan
                continue

            stack = np.vstack([s.values for s in member_series])
            count_vals = np.sum(~np.isnan(stack), axis=0)
            with np.errstate(invalid="ignore"):
                mean_vals = np.nanmean(stack, axis=0)
            mean_vals[count_vals < min_coverage] = np.nan
            avg_df[col] = mean_vals

        avg_df = avg_df.dropna(subset=cols, how="all").reset_index(drop=True)
        return DataStream(avg_df), {"n_members": len(data_streams)}

    # ========== Technique 0: average-ensemble statistics ==========

    def _tech0_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        avg_ds: Optional[DataStream] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute Technique-0 statistics for one column.

        Uses a pre-computed average-ensemble DataStream (or computes it).
        Runs DataStream.compute_statistics() on the single averaged trace.
        """
        if avg_ds is None:
            avg_ds = self.compute_average_ensemble()

        if col not in avg_ds.data.columns:
            out = {"mean": np.nan, "mean_uncertainty": np.nan,
                   "confidence_interval": (np.nan, np.nan), "pm_std": (np.nan, np.nan),
                   "warning": f"Column '{col}' missing from average ensemble."}
            return out, {}

        s = avg_ds.compute_statistics(column_name=col, ddof=ddof, method=method, window_size=window_size)
        stat = s.get(col, {}) if isinstance(s, dict) else {}
        return stat, {"n_members_averaged": len(self.data_streams)}

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
        """
        For one member: choose window so block means pass Ljung-Box, or use
        the best-achievable window. Returns (block_means_array, meta_dict).
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

        if col not in ds.data.columns:
            out_meta["status"] = "best_p"
            out_meta["warnings"].append("column_missing")
            return np.array([], dtype=float), out_meta

        series = ds.data[col].dropna()
        n = int(series.size)
        if n < 4:
            out_meta["status"] = "best_p"
            out_meta["warnings"].append("too_few_samples")
            return np.array([], dtype=float), out_meta

        w = int(max(2, ds._estimate_window(col, series, window_size)))
        best_blocks = np.array([], dtype=float)

        for trial in range(max_tries):
            out_meta["tries"] = trial + 1
            if w >= n:
                break

            time_values = ds._time_values_for_series(series)
            bm_series = ds._process_column(series, w, method, time_values=time_values)
            if bm_series is None or len(bm_series) == 0:
                w += 1
                continue

            blocks = np.asarray(bm_series.values, dtype=float)
            n_blocks = int(blocks.size)

            if n_blocks < max(3, min_blocks):
                if best_blocks.size == 0:
                    best_blocks = blocks
                    out_meta["best_window"] = int(w)
                out_meta["warnings"].append("too_few_blocks_for_ljungbox")
                break

            lags = lb_lags if lb_lags is not None else max(1, min(20, n_blocks // 4))
            try:
                lb = acorr_ljungbox(blocks, lags=[lags], return_df=True)
                pval = float(lb["lb_pvalue"].iloc[0])
            except Exception:
                pval = np.nan

            if np.isfinite(pval) and pval > float(out_meta["best_pvalue"]):
                out_meta["best_pvalue"] = pval
                out_meta["best_window"] = int(w)
                best_blocks = blocks

            if np.isfinite(pval) and pval >= lb_alpha:
                out_meta.update({
                    "status": "independent",
                    "window_used": int(w),
                    "n_blocks": int(n_blocks),
                    "ljungbox_pvalue": float(pval),
                    "ljungbox_lags": int(lags),
                })
                return blocks, out_meta

            w += 1

        if best_blocks.size > 0 and out_meta["best_window"] is not None:
            n_blocks = int(best_blocks.size)
            lags = lb_lags if lb_lags is not None else max(1, min(20, n_blocks // 4))
            try:
                lb = acorr_ljungbox(best_blocks, lags=[lags], return_df=True)
                pval = float(lb["lb_pvalue"].iloc[0])
            except Exception:
                pval = np.nan
            out_meta.update({
                "status": "best_p",
                "window_used": int(out_meta["best_window"]),
                "n_blocks": int(n_blocks),
                "ljungbox_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "ljungbox_lags": int(lags),
            })
            out_meta["warnings"].append("independence_failed_used_best_p_window")
            return best_blocks, out_meta

        out_meta["status"] = "best_p"
        out_meta["warnings"].append("no_valid_blocks")
        return np.array([], dtype=float), out_meta

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
        """
        For all members: autotune per-member window, pool block means.
        Returns (pooled_blocks, meta).
        """
        pooled: List[np.ndarray] = []
        meta: Dict[str, Any] = {
            "members_used": 0,
            "member_windows": [],
            "member_blocks": [],
            "member_status": [],
            "member_pvalues": [],
            "member_lags": [],
            "member_warnings": [],
        }

        for ds in self.data_streams:
            blocks, m = self._autotune_member_blocks_until_independent(
                ds=ds, col=col, window_size=window_size, method=method,
                lb_alpha=lb_alpha, lb_lags=lb_lags, max_tries=max_tries, min_blocks=min_blocks,
            )
            if blocks is None or blocks.size == 0:
                continue

            pooled.append(blocks.astype(float, copy=False))
            meta["members_used"] += 1
            meta["member_windows"].append(int(m.get("window_used") or -1))
            meta["member_blocks"].append(int(m.get("n_blocks", 0) or 0))
            meta["member_status"].append(m.get("status") or "unknown")
            meta["member_pvalues"].append(float(m.get("ljungbox_pvalue", np.nan)))
            meta["member_lags"].append(int(m.get("ljungbox_lags", 0) or 0))
            meta["member_warnings"].append(m.get("warnings", []))

        if not pooled:
            return np.array([], dtype=float), meta

        return np.concatenate(pooled, axis=0), meta

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
        Compute Technique-1 (pooled-block) statistics for one column.

        Policy:
        - Always compute sem_n (sd/sqrt(n_blocks)) and sem_ess (sd/sqrt(ESS_blocks)).
        - Default reported SEM = sem_n.
        - If pooled Ljung-Box p-value < pooled_lb_alpha_bad, switch to sem_ess.
        """
        pooled, meta = self._pooled_block_means_from_trimmed(
            col=col, window_size=window_size, method="non-overlapping",
            lb_alpha=lb_alpha, lb_lags=lb_lags,
        )

        member_status = meta.get("member_status", [])
        all_independent = bool(member_status) and all(s == "independent" for s in member_status)
        some_best_p = any(s == "best_p" for s in member_status)

        out: Dict[str, Any] = {}
        if pooled.size == 0:
            out.update({
                "mean": np.nan, "variance": np.nan, "ess_blocks": np.nan,
                "n_short_averages": 0, "mean_uncertainty": np.nan,
                "mean_uncertainty_sem_n": np.nan, "mean_uncertainty_sem_ess": np.nan,
                "se_method": None, "warning": "No pooled block means available.",
                "confidence_interval": (np.nan, np.nan), "pm_std": (np.nan, np.nan),
                "independent": None, "ljungbox_pvalue": np.nan, "ljungbox_lags": 0,
                "member_all_independent": all_independent, "member_some_best_p": some_best_p,
            })
            meta["pooled_blocks"] = 0
            return out, meta

        n_blocks = int(pooled.size)
        mu = float(np.mean(pooled))
        var_blocks = float(np.var(pooled, ddof=ddof)) if n_blocks > 1 else 0.0
        ess_blocks = float(max(1.0, _geyer_ess_on_blocks(pooled)))
        sem_n = float(np.sqrt(var_blocks) / np.sqrt(max(1.0, float(n_blocks))))
        sem_ess = float(np.sqrt(var_blocks) / np.sqrt(ess_blocks))

        lags = lb_lags if lb_lags is not None else int(max(1, min(20, n_blocks // 4)))
        try:
            lb = acorr_ljungbox(pooled, lags=[lags], return_df=True)
            pooled_pval = float(lb["lb_pvalue"].iloc[0])
            pooled_independent = bool(pooled_pval >= lb_alpha)
        except Exception:
            pooled_pval = np.nan
            pooled_independent = None

        sem_reported = sem_n
        se_method = "sem_n"
        warning = "Some members used best_p window." if some_best_p else None

        if np.isfinite(pooled_pval) and pooled_pval < float(pooled_lb_alpha_bad):
            sem_reported = sem_ess
            se_method = "sem_ess (pooled_LB_bad)"
            extra = f"Pooled LB p={pooled_pval:.3g} < {pooled_lb_alpha_bad}; switched to ESS-based SEM."
            warning = extra if warning is None else (warning + " " + extra)

        ci = (float(mu - 1.96 * sem_reported), float(mu + 1.96 * sem_reported))
        pm_std = (float(mu - sem_reported), float(mu + sem_reported))

        out.update({
            "mean": mu,
            "variance": var_blocks,
            "ess_blocks": ess_blocks,
            "n_short_averages": n_blocks,
            "mean_uncertainty": sem_reported,
            "mean_uncertainty_sem_n": sem_n,
            "mean_uncertainty_sem_ess": sem_ess,
            "se_method": se_method,
            "warning": warning,
            "confidence_interval": ci,
            "pm_std": pm_std,
            "independent": pooled_independent,
            "ljungbox_pvalue": pooled_pval,
            "ljungbox_lags": int(lags),
            "member_all_independent": all_independent,
            "member_some_best_p": some_best_p,
        })
        meta["pooled_blocks"] = n_blocks
        return out, meta

    # ========== Technique 2: member-wise then aggregate ==========

    def _tech2_stats_for_col(
        self,
        col: str,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        diagnostics: str = "compact",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute Technique-2 statistics for one column.
        Calls DataStream.compute_statistics() on each member and aggregates via
        inverse-variance weighting (fallback: simple mean).
        """
        indiv: Dict[str, Any] = {}
        means, ses, w_means, weights = [], [], [], []
        var_list, ess_list, nsa_list = [], [], []

        for i, ds in enumerate(self.data_streams):
            key = f"Member {i}"
            if col not in ds.data.columns:
                continue
            s = ds.compute_statistics(column_name=col, ddof=ddof, method=method, window_size=window_size)
            stat = s.get(col, {}) if isinstance(s, dict) else {}
            if not isinstance(stat, dict):
                continue

            mu = stat.get("mean", np.nan)
            se = stat.get("mean_uncertainty", np.nan)
            ci = stat.get("confidence_interval", (np.nan, np.nan))
            pm = stat.get("pm_std", (np.nan, np.nan))
            ess_i = stat.get("ess_blocks", stat.get("block_effective_n", np.nan))
            nsa_i = stat.get("n_short_averages", np.nan)
            var_i = stat.get("variance", np.nan)

            indiv[key] = {
                "mean": mu, "mean_uncertainty": se, "variance": var_i,
                "confidence_interval": ci, "pm_std": pm,
                "ess_blocks": ess_i, "n_short_averages": nsa_i,
            }
            if np.isfinite(mu):
                means.append(float(mu))
            if np.isfinite(se):
                ses.append(float(se))
            if np.isfinite(mu) and np.isfinite(se) and se > 0:
                w_means.append(float(mu))
                weights.append(1.0 / float(se) ** 2)
            if np.isfinite(var_i):
                var_list.append(float(var_i))
            if np.isfinite(ess_i):
                ess_list.append(float(ess_i))
            if np.isfinite(nsa_i):
                nsa_list.append(float(nsa_i))

        if weights and w_means:
            w = np.asarray(weights, dtype=float)
            m = np.asarray(w_means, dtype=float)
            w_sum = float(np.sum(w))
            mu_ens = float(np.sum(w * m) / w_sum) if w_sum > 0 else np.nan
            se_ens = float(np.sqrt(1.0 / w_sum)) if w_sum > 0 else np.nan
        else:
            mu_ens = float(np.nanmean(means)) if means else np.nan
            se_ens = float(np.nanmean(ses)) if ses else np.nan

        var_ens = float(np.nanmean(var_list)) if var_list else np.nan
        ess_ens = float(np.nanmean(ess_list)) if ess_list else np.nan
        nsa_ens = float(np.nanmean(nsa_list)) if nsa_list else np.nan
        ci_ens = (
            float(mu_ens - 1.96 * se_ens) if np.isfinite(se_ens) else np.nan,
            float(mu_ens + 1.96 * se_ens) if np.isfinite(se_ens) else np.nan,
        )

        out: Dict[str, Any] = {
            "mean": mu_ens,
            "mean_uncertainty": se_ens,
            "confidence_interval": ci_ens,
            "pm_std": (
                (mu_ens - se_ens) if np.isfinite(mu_ens) and np.isfinite(se_ens) else np.nan,
                (mu_ens + se_ens) if np.isfinite(mu_ens) and np.isfinite(se_ens) else np.nan,
            ),
            "variance": var_ens,
            "ess_blocks": ess_ens,
            "n_short_averages": nsa_ens,
            "se_method": "tech2_inverse_variance_weighted",
            "warning": None,
            "individual": indiv if diagnostics == "full" else None,
        }
        meta: Dict[str, Any] = {"n_members_used": len(indiv)}
        return out, meta

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
        from .trim import build_trim_strategy, TrimDataStreamOperation

        # Backward-compat: accept batch_size as alias for window_size
        if "batch_size" in kwargs:
            window_size = kwargs.pop("batch_size")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

        strategy = build_trim_strategy(
            method=method,
            window_size=window_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )
        op = TrimDataStreamOperation(strategy=strategy)
        trimmed = [op(ds, column_name=column_name) for ds in self.data_streams]
        kept = [t for t in trimmed if t is not None and not t.data.empty]
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
            r = ds.is_stationary(columns)  # simple {col: bool} from DataStream
            results[f"Member {i}"] = r
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
        """
        cols = self._resolve_cols(column_name)
        stats: Dict = {}
        metadata: Dict = {}

        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            meta_cols = {}
            for col in cols:
                col_stats, col_meta = self._tech0_stats_for_col(
                    col=col, ddof=ddof, method=method, window_size=window_size, avg_ds=avg_ds,
                )
                stats[col] = col_stats
                meta_cols[col] = col_meta
            metadata["technique_0_average_ensemble"] = meta_cols
            return {"results": stats, "metadata": metadata}

        elif technique == 1:
            meta_cols = {}
            for col in cols:
                col_stats, col_meta = self._tech1_pooled_stats_for_col(
                    col=col, ddof=ddof, window_size=window_size, method=method,
                )
                stats[col] = col_stats
                meta_cols[col] = col_meta
            metadata["technique_1_pooled"] = meta_cols
            return {"results": stats, "metadata": metadata}

        elif technique == 2:
            meta_cols = {}
            for col in cols:
                col_stats, col_meta = self._tech2_stats_for_col(
                    col=col, ddof=ddof, method=method, window_size=window_size, diagnostics=diagnostics,
                )
                stats[col] = {
                    "mean": col_stats["mean"],
                    "mean_uncertainty": col_stats["mean_uncertainty"],
                    "confidence_interval": col_stats["confidence_interval"],
                    "pm_std": col_stats["pm_std"],
                    "variance": col_stats.get("variance", np.nan),
                    "ess_blocks": col_stats.get("ess_blocks", np.nan),
                    "n_short_averages": col_stats.get("n_short_averages", np.nan),
                    "se_method": col_stats.get("se_method"),
                    "warning": col_stats.get("warning"),
                    "individual": col_stats.get("individual") if diagnostics == "full" else None,
                }
                meta_cols[col] = col_meta
            metadata["technique_2_memberwise"] = meta_cols
            return {"results": stats, "metadata": metadata}

        else:
            raise ValueError("Invalid technique. Use 0, 1, or 2.")

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
