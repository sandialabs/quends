# ensemble.py

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
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
        """
        Variables present in ALL members (excluding 'time').
        """
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

    # ---------- Helpers for pooled block means (Technique 1) ----------
    @staticmethod
    def _geyer_ess_on_blocks(block_means: np.ndarray) -> float:
        """
        Geyer positive-pair ESS on an already block-meaned series.
        """
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
        ess = max(1.0, n / (1.0 + 2.0 * s))
        return ess

    def _pooled_block_means_from_trimmed(
        self,
        col: str,
        window_size: Optional[int],
        method: str = "non-overlapping",
    ) -> np.ndarray:
        """
        Build pooled block means for an already-trimmed Ensemble.

        For each member:
          - take the (trimmed) series for `col`
          - estimate per-member block length with ds._estimate_window(col, series, window_size)
          - compute block means ONCE using ds._process_column(..., method="non-overlapping")

        Returns
        -------
        1-D numpy array of pooled block means across members (may be empty).
        """
        pooled: List[np.ndarray] = []

        for ds in self.data_streams:
            if col not in ds.df.columns:
                continue
            series = ds.df[col].dropna()
            if series.size < 2:
                continue

            est_win = ds._estimate_window(col, series, window_size)
            if est_win <= 1 or series.size < est_win:
                continue

            bm = ds._process_column(
                series, estimated_window=est_win, method=method
            )
            if bm is not None and len(bm) > 0:
                pooled.append(np.asarray(bm.values, dtype=float))

        if not pooled:
            return np.array([], dtype=float)
        return np.concatenate(pooled, axis=0)

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

    # ---------- Classic ESS across ensemble (unchanged techniques) ----------
    def effective_sample_size(
        self,
        column_names: Optional[str | List[str]] = None,
        alpha: float = 0.05,
        technique: int = 0,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Classic ESS via three techniques:
          0 - on average-ensemble (interpolated grid mean)
          1 - on concatenated aggregate of raw series (per variable)
          2 - per-member then aggregate (mean of ESS values)

        diagnostics: 'none' | 'compact' | 'full'
        """
        metadata: Dict = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = avg_ds.effective_sample_size(
                column_names=column_names,
                method="geyer",
                alpha=alpha,
                diagnostics="none",
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names]
                if isinstance(column_names, str)
                else (
                    self.common_variables()
                    if column_names is None
                    else list(column_names)
                )
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = ds_agg.effective_sample_size(
                    column_names=list(agg_df.columns),
                    method="geyer",
                    alpha=alpha,
                    diagnostics="none",
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            per_member_results, per_member_meta = {}, {}
            for i, ds in enumerate(self.data_streams):
                res = ds.effective_sample_size(
                    column_names=column_names,
                    method="geyer",
                    alpha=alpha,
                    diagnostics="none",
                )
                per_member_results[f"Member {i}"] = (
                    res.get("results") if isinstance(res, dict) else res
                )
                if diagnostics == "full":
                    per_member_meta[f"Member {i}"] = getattr(
                        ds, "_history", None
                    )
            ess_vals = []
            for v in per_member_results.values():
                if isinstance(v, dict):
                    for ess in v.values():
                        if isinstance(ess, (int, float)) and not np.isnan(ess):
                            ess_vals.append(ess)
            agg_ess = np.nanmean(ess_vals) if ess_vals else np.nan
            result = {
                "ensemble_ess": agg_ess,
                "individual_ess": per_member_results,
            }
            if diagnostics == "full":
                metadata["per_member"] = per_member_meta
        else:
            raise ValueError("Invalid technique. Choose 0, 1, or 2.")

        return {"results": result, "metadata": metadata}

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
        if technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else (
                    self.common_variables()
                    if column_name is None
                    else list(column_name)
                )
            )
            result: Dict = {}
            meta_cols: Dict = {}
            for col in cols:
                pooled = self._pooled_block_means_from_trimmed(
                    col=col, window_size=window_size, method=method
                )
                if pooled.size == 0:
                    result[col] = np.nan
                    meta_cols[col] = {"pooled_blocks": 0}
                    continue
                mu = float(np.mean(pooled))
                result[col] = mu
                meta_cols[col] = {"pooled_blocks": int(pooled.size)}
            metadata["pooled_blocks"] = meta_cols

        elif technique == 2:
            # member-wise mean then aggregate
            member_means: Dict[str, Dict[str, float]] = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                stats_result = ds.compute_statistics(
                    column_name=column_name,
                    method=method,
                    window_size=window_size,
                    diagnostics="none",
                )
                mean_subdict = {}
                for col, stat in stats_result.items():
                    if isinstance(stat, dict) and "mean" in stat:
                        mean_subdict[col] = stat["mean"]
                member_means[key] = mean_subdict

            agg_cols = (
                self.common_variables()
                if column_name is None
                else (
                    [column_name]
                    if isinstance(column_name, str)
                    else list(column_name)
                )
            )
            ensemble_mean = {}
            for col in agg_cols:
                values = [
                    member_means[k][col]
                    for k in member_means
                    if col in member_means[k]
                ]
                if values:
                    ensemble_mean[col] = float(np.mean(values))

            if diagnostics == "full":
                metadata["individual_histories"] = {
                    f"Member {i}": getattr(ds, "_history", None)
                    for i, ds in enumerate(self.data_streams)
                }

            result = {
                "Member Ensemble": ensemble_mean,
                "Individual Members": member_means,
            }
        else:
            raise ValueError("Invalid technique. Use 1 or 2.")
        return {"results": result, "metadata": metadata}

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
        Ensemble SEM.

        Techniques:
          1 - pooled block-means from already-trimmed members (preferred)
              SEM = std(pooled, ddof)/sqrt(ESS_blocks(Geyer))
          2 - member-wise SEM then aggregate
        """
        metadata: Dict = {}
        if technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else (
                    self.common_variables()
                    if column_name is None
                    else list(column_name)
                )
            )
            result: Dict = {}
            meta_cols: Dict = {}
            for col in cols:
                pooled = self._pooled_block_means_from_trimmed(
                    col=col, window_size=window_size, method=method
                )
                if pooled.size == 0:
                    result[col] = np.nan
                    meta_cols[col] = {"pooled_blocks": 0, "ess_blocks": np.nan}
                    continue
                ess_blocks = self._geyer_ess_on_blocks(pooled)
                sem = float(np.std(pooled, ddof=ddof) / np.sqrt(ess_blocks))
                result[col] = sem
                meta_cols[col] = {
                    "pooled_blocks": int(pooled.size),
                    "ess_blocks": float(ess_blocks),
                }
            metadata["pooled_blocks"] = meta_cols

        elif technique == 2:
            member_unc: Dict[str, Dict[str, float]] = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                stats_result = ds.compute_statistics(
                    column_name=column_name,
                    method=method,
                    window_size=window_size,
                    diagnostics="none",
                )
                unc_subdict = {}
                for col, stat in stats_result.items():
                    if isinstance(stat, dict) and "mean_uncertainty" in stat:
                        unc_subdict[col] = stat["mean_uncertainty"]
                member_unc[key] = unc_subdict

            agg_cols = (
                self.common_variables()
                if column_name is None
                else (
                    [column_name]
                    if isinstance(column_name, str)
                    else list(column_name)
                )
            )
            ensemble_unc = {}
            for col in agg_cols:
                values = [
                    member_unc[k][col]
                    for k in member_unc
                    if col in member_unc[k]
                ]
                if values:
                    ensemble_unc[col] = float(np.mean(values))

            if diagnostics == "full":
                metadata["individual_histories"] = {
                    f"Member {i}": getattr(ds, "_history", None)
                    for i, ds in enumerate(self.data_streams)
                }

            result = {
                "Member Ensemble": ensemble_unc,
                "Individual Members": member_unc,
            }
        else:
            raise ValueError("Invalid technique. Use 1 or 2.")
        return {"results": result, "metadata": metadata}

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

        Techniques:
          1 - pooled block-means from already-trimmed members (preferred)
              CI = mean ± 1.96*SEM (SEM as defined in mean_uncertainty technique 1)
          2 - member-wise CI then aggregate (average of bounds)
        """
        metadata: Dict = {}
        if technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else (
                    self.common_variables()
                    if column_name is None
                    else list(column_name)
                )
            )
            result: Dict = {}
            meta_cols: Dict = {}
            for col in cols:
                pooled = self._pooled_block_means_from_trimmed(
                    col=col, window_size=window_size, method=method
                )
                if pooled.size == 0:
                    result[col] = (np.nan, np.nan)
                    meta_cols[col] = {"pooled_blocks": 0, "ess_blocks": np.nan}
                    continue
                mu = float(np.mean(pooled))
                ess_blocks = self._geyer_ess_on_blocks(pooled)
                sem = float(np.std(pooled, ddof=ddof) / np.sqrt(ess_blocks))
                ci = (float(mu - 1.96 * sem), float(mu + 1.96 * sem))
                result[col] = ci
                meta_cols[col] = {
                    "pooled_blocks": int(pooled.size),
                    "ess_blocks": float(ess_blocks),
                    "mean": mu,
                    "sem": sem,
                }
            metadata["pooled_blocks"] = meta_cols

        elif technique == 2:
            member_cis: Dict[str, Dict[str, Tuple[float, float]]] = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                stats_result = ds.compute_statistics(
                    column_name=column_name,
                    method=method,
                    window_size=window_size,
                    diagnostics="none",
                )
                ci_subdict = {}
                for col, stat in stats_result.items():
                    if isinstance(stat, dict) and "confidence_interval" in stat:
                        ci_subdict[col] = stat["confidence_interval"]
                member_cis[key] = ci_subdict

            agg_cols = (
                self.common_variables()
                if column_name is None
                else (
                    [column_name]
                    if isinstance(column_name, str)
                    else list(column_name)
                )
            )
            ensemble_ci = {}
            for col in agg_cols:
                vals = [
                    member_cis[k][col]
                    for k in member_cis
                    if col in member_cis[k]
                ]
                if vals and all(
                    isinstance(v, (tuple, list, np.ndarray)) and len(v) == 2
                    for v in vals
                ):
                    lower = float(np.mean([v[0] for v in vals]))
                    upper = float(np.mean([v[1] for v in vals]))
                    ensemble_ci[col] = (lower, upper)
                else:
                    ensemble_ci[col] = (np.nan, np.nan) if not vals else None

            if diagnostics == "full":
                metadata["individual_histories"] = {
                    f"Member {i}": getattr(ds, "_history", None)
                    for i, ds in enumerate(self.data_streams)
                }

            result = {
                "Member Ensemble": ensemble_ci,
                "Individual Members": member_cis,
            }
        else:
            raise ValueError("Invalid technique. Use 1 or 2.")
        return {"results": result, "metadata": metadata}

    def compute_statistics(
        self,
        column_name: Optional[str | List[str]] = None,
        ddof: int = 1,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
        technique: int = 1,
        diagnostics: str = "compact",
    ) -> Dict:
        """
        Aggregate mean, SEM, CI, and ±1*SEM band across the ensemble.

        diagnostics: 'none' | 'compact' | 'full'
        """
        mean_result = self.mean(
            column_name, method, window_size, technique, diagnostics
        )
        unc_result = self.mean_uncertainty(
            column_name, ddof, method, window_size, technique, diagnostics
        )
        ci_result = self.confidence_interval(
            column_name, ddof, method, window_size, technique, diagnostics
        )

        stats: Dict = {}
        if technique == 2:
            for key in mean_result["results"]["Member Ensemble"]:
                mu = mean_result["results"]["Member Ensemble"][key]
                se = unc_result["results"]["Member Ensemble"].get(key, np.nan)
                ci = ci_result["results"]["Member Ensemble"].get(
                    key, (np.nan, np.nan)
                )
                stats[key] = {
                    "mean": mu,
                    "mean_uncertainty": se,
                    "confidence_interval": ci,
                    "pm_std": (
                        mu - se if np.isfinite(se) else np.nan,
                        mu + se if np.isfinite(se) else np.nan,
                    ),
                }
        else:
            # technique 1 returns flat dict {col: value}
            keys = mean_result["results"].keys()
            for key in keys:
                mu = mean_result["results"].get(key, np.nan)
                se = unc_result["results"].get(key, np.nan)
                ci = ci_result["results"].get(key, (np.nan, np.nan))
                stats[key] = {
                    "mean": mu,
                    "mean_uncertainty": se,
                    "confidence_interval": ci,
                    "pm_std": (
                        mu - se if np.isfinite(se) else np.nan,
                        mu + se if np.isfinite(se) else np.nan,
                    ),
                }

        metadata = {
            "mean": mean_result["metadata"],
            "mean_uncertainty": unc_result["metadata"],
            "confidence_interval": ci_result["metadata"],
        }
        return {"results": stats, "metadata": metadata}

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
