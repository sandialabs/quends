import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, rankdata
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf, adfuller

from quends.base.utils import power_law_model

from .history import DataStreamHistory
from .utils import (
    _compute_ess,
    _geyer_ess_on_blocks,
    _ljung_box_pass,
    _resolve_columns,
    _tau_int_geyer_from_acf,
    to_native_types,
)

"""
Usage signature for the new trimming workflow::

    strategy = SomeTrimStrategy(kwargs...)
    trim_operation = TrimDataStreamOperation(strategy=strategy)
    data_stream = DataStream(data)
    trimmed_data_stream = trim_operation(data_stream, column_name)
"""


# Goes into datastream
class DataStream:

    def __init__(self, data: Any, history: Optional[DataStreamHistory] = None) -> None:
        self._data = data
        self._history = history or DataStreamHistory()

    @property
    def data(self) -> Any:
        """The underlying pandas DataFrame."""
        return self._data

    @property
    def history(self) -> DataStreamHistory:
        return self._history

    def head(self, n: int = 5) -> Any:
        return self.data.head(n)

    def __len__(self) -> int:
        return len(self.data)

    def variables(self):
        """
        List the signal variable (column) names, excluding the 'time' column.

        Returns
        -------
        pandas.Index
            Column names in ``self.data``.
        """
        return self.data.columns

    # --------- Statistical summaries ---------
    def mean(self, column_name=None, method="non-overlapping", window_size=None):
        """Compute block or sliding window means for each column."""
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.data[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            time_values = self._time_values_for_series(column_data)
            proc_data = self._process_column(column_data, est_win, method, time_values=time_values)
            results[col] = {
                "mean": float(np.mean(proc_data)),
                "window_size": int(est_win),
            }
        return results

    def mean_uncertainty(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """Estimate the standard error of the mean via Geyer ESS on block means."""
        results = {}
        for col in self._get_columns(column_name):
            series = self.data[col].dropna()
            if series.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            info = self.get_block_effective_n(col, method=method, window_size=window_size)
            est_win = info["window_size"]
            eff_n = max(1.0, info["effective_n"] if np.isfinite(info["effective_n"]) else 1.0)
            time_values = self._time_values_for_series(series)
            proc_data = self._process_column(series, est_win, method, time_values=time_values)
            uncertainty = float(np.std(proc_data.values, ddof=ddof) / np.sqrt(eff_n))
            results[col] = {
                "mean_uncertainty": uncertainty,
                "window_size": int(est_win),
            }
        return results

    def confidence_interval(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Build 95% confidence intervals around block/sliding means.

        Private helper.
        """
        results = {}
        mean_results = self.mean(column_name, method=method, window_size=window_size)
        uncertainty_results = self.mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        for col in self._get_columns(column_name):
            if col not in mean_results or col not in uncertainty_results:
                results[col] = {"error": f"Missing data for column '{col}'"}
                continue
            mean_val = mean_results[col]["mean"]
            uncertainty_val = uncertainty_results[col]["mean_uncertainty"]
            ci_lower = mean_val - 1.96 * uncertainty_val
            ci_upper = mean_val + 1.96 * uncertainty_val
            results[col] = {
                "confidence_interval": (float(ci_lower), float(ci_upper)),
                "window_size": int(mean_results[col]["window_size"]),
            }
        return results

    def compute_statistics(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Aggregate statistics: mean, uncertainty, CI, pm_std bounds, ESS, and window size.

        Appends the operation to history and embeds deduplicated metadata in the results.

        Parameters
        ----------
        column_name : str or list or None
        ddof : int
        method : {'sliding', 'non-overlapping'}
        window_size : int or None

        Returns
        -------
        dict
            {col: {statistics...}, 'metadata': history}
        """
        stats = {}
        cols = self._get_columns(column_name)
        ess_res = self.effective_sample_size(column_names=column_name)

        for col in cols:
            series = self.data[col].dropna()
            if series.empty:
                stats[col] = {"error": f"No data available for column '{col}'"}
                continue

            # Window selection with independence attempt
            if window_size is None:
                w, info = self._autotune_window_size(
                    col=col, series=series, method=method,
                    alpha=0.05, lag_set=(5, 10), c0=2.0, max_iter=20, w_min=5,
                )
                status = info.get("status", "best_p")
                lb = info.get("lb", {"lags": [], "pvalues": []})
            else:
                w = int(window_size)
                status = "user_window"
                lb = {"lags": [], "pvalues": []}

            time_values = self._time_values_for_series(series)
            proc = self._process_column(series, w, method, time_values=time_values)
            block_means = np.asarray(proc.values, dtype=float)
            n_blocks = int(block_means.size)

            if n_blocks < 1:
                stats[col] = {"error": f"No block means produced (window_size={w}).", "window_size": int(w)}
                continue

            mu = float(np.mean(block_means))
            var_val = float(np.var(block_means, ddof=ddof)) if n_blocks >= 2 else np.nan
            sd = float(np.std(block_means, ddof=ddof)) if n_blocks >= 2 else np.nan
            ess_blocks = _geyer_ess_on_blocks(block_means)

            # SE rule based on independence status
            warning = None
            if status == "independent":
                eff_n_for_se = float(n_blocks)
                se_method = "iid_blocks"
            elif status == "best_p":
                eff_n_for_se = float(max(1.0, n_blocks))
                se_method = "iid_blocks_best_p"
                warning = "Block means did not pass Ljung-Box; using best-p window."
            elif status == "user_window":
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks"
                warning = "User window; SE via Geyer ESS on block means."
            else:
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks_fallback"
                warning = "Too few blocks for independence test; SE via Geyer ESS."

            se = float(sd / np.sqrt(eff_n_for_se)) if np.isfinite(sd) and n_blocks >= 2 else np.nan
            ci = (float(mu - 1.96 * se), float(mu + 1.96 * se)) if np.isfinite(se) else (np.nan, np.nan)

            ess_entry = ess_res.get("results", {}).get(col, {})
            ess_val = ess_entry.get("effective_sample_size") if isinstance(ess_entry, dict) else ess_entry

            entry = {
                "mean": mu,
                "mean_uncertainty": se,
                "variance": var_val,
                "confidence_interval": ci,
                "pm_std": (mu - se, mu + se) if np.isfinite(se) else (np.nan, np.nan),
                "effective_sample_size": int(ess_val) if ess_val is not None else None,
                "window_size": int(w),
                "n_short_averages": int(n_blocks),
                "ess_blocks": float(ess_blocks),
                "block_effective_n": float(ess_blocks),
                "se_effective_n": float(eff_n_for_se),
                "se_method": se_method,
                "independence_status": status,
                "ljungbox_lags": lb.get("lags", []),
                "ljungbox_pvalues": lb.get("pvalues", []),
            }
            if warning:
                entry["warning"] = warning
            stats[col] = entry

        return to_native_types(stats)

    def cumulative_statistics(
        self, column_name=None, method="non-overlapping", window_size=None
    ):
        """
        Generate cumulative mean and uncertainty time series for each column.

        Records operation and returns per-column cumulative arrays plus window_size.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.data[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            time_values = self._time_values_for_series(column_data)
            proc_data = self._process_column(column_data, est_win, method, time_values=time_values)
            cumulative_mean = proc_data.expanding().mean()
            cumulative_std = proc_data.expanding().std()
            count = proc_data.expanding().count()
            standard_error = cumulative_std / np.sqrt(count)
            results[col] = {
                "cumulative_mean": cumulative_mean.tolist(),
                "cumulative_uncertainty": cumulative_std.tolist(),
                "standard_error": standard_error.tolist(),
                "window_size": int(est_win),
            }

        return to_native_types(results)

    def additional_data(
        self,
        column_name=None,
        ddof=1,
        method="sliding",
        window_size=None,
        reduction_factor=0.1,
    ):
        """
        Estimate additional sample size needed to reduce SEM by `reduction_factor` via power-law fit.

        Records operation and returns model parameters and sample projections.
        """
        stats = self.cumulative_statistics(
            column_name, method=method, window_size=window_size
        )
        results = {}
        columns = self._get_columns(column_name)
        for col in columns:
            if "cumulative_uncertainty" not in stats.get(col, {}):
                results[col] = {"error": f"No cumulative SEM data for column '{col}'"}
                continue
            column_data = self.data[col].dropna()
            est_win = self._estimate_window(col, column_data, window_size)
            cum_sem = np.array(stats[col]["cumulative_uncertainty"])
            n_current = len(cum_sem)
            cumulative_count = np.arange(1, n_current + 1)
            mask = np.isfinite(cum_sem)
            valid_count = cumulative_count[mask]
            valid_sem = cum_sem[mask]
            if len(valid_count) < 2:
                results[col] = {"error": "Not enough valid data points for fitting."}
                continue

            popt, _ = curve_fit(power_law_model, valid_count, valid_sem, p0=[1.0, 0.5])
            A_est, p_est = popt
            p_est = abs(p_est)
            current_sem = power_law_model(n_current, A_est, p_est)
            target_sem = (1 - reduction_factor) * current_sem
            n_target = (A_est / target_sem) ** (1 / p_est)
            additional_samples = n_target - n_current
            if method == "non-overlapping":
                additional_samples *= est_win
            results[col] = {
                "A_est": float(A_est),
                "p_est": float(p_est),
                "n_current": int(n_current),
                "current_sem": float(current_sem),
                "target_sem": float(target_sem),
                "n_target": float(n_target),
                "additional_samples": int(math.ceil(additional_samples)),
                "window_size": int(est_win),
            }

        return to_native_types(results)

    def effective_sample_size_below(self, column_names=None, alpha=0.05):
        """
        Stub for compatibility with legacy test. Returns dummy value.
        """
        # We could implement a real one if needed; for now, return 0 for all columns.
        if column_names is None:
            column_names = [col for col in self.data.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
        return {col: 0 for col in column_names}

    # ------ Helper functions --------
    def is_stationary(self, columns):
        """
        Perform Augmented Dickey-Fuller test for each specified column.

        Parameters
        ----------
        columns : str or list of str

        Returns
        -------
        dict
            {column: True if stationary (p < 0.05), else False}
        """
        if isinstance(columns, str):
            columns = [columns]
        results = {}
        for column in columns:
            try:
                p_value = adfuller(self.data[column].dropna(), autolag="AIC")[1]
                results[column] = bool(p_value < 0.05)
            except Exception as e:
                results[column] = False
        return results

    def _get_columns(self, column_name):
        """
        Resolve `column_name` parameter into a list of valid DataFrame columns.

        Returns
        -------
        list of str
        """
        if column_name is None:
            return [col for col in self.data.columns if col != "time"]
        return [column_name] if isinstance(column_name, str) else column_name

    def _estimate_window(self, col, column_data, window_size):
        """
        Determine block size: either user-provided or via tau_int + Ljung-Box autotune.

        Ensures a minimum window of 5 samples.
        """
        if window_size is not None:
            return int(window_size)
        w, _ = self._autotune_window_size(
            col=col,
            series=column_data,
            method="non-overlapping",
            alpha=0.05,
            lag_set=(5, 10),
            B_min=15,
            c0=2.0,
            max_iter=20,
            w_min=5,
        )
        return int(w)

    def _process_column(
        self,
        column_data,
        estimated_window,
        method,
        time_values: Optional[np.ndarray] = None,
    ):
        """
        Transform a 1D series into block or sliding window means.

        Parameters
        ----------
        column_data : pandas.Series
        estimated_window : int
        method : {'sliding', 'non-overlapping'}
        time_values : np.ndarray or None
            If provided, used as block-center indices for non-overlapping means.

        Returns
        -------
        pandas.Series
        """
        if time_values is not None:
            time_values = np.asarray(time_values, dtype=float)
            if time_values.size != len(column_data):
                time_values = None

        if method == "sliding":
            if time_values is not None:
                series = pd.Series(column_data.values, index=time_values)
            else:
                series = column_data
            return series.rolling(window=int(estimated_window)).mean().dropna()

        elif method == "non-overlapping":
            w = int(max(1, estimated_window))
            x = np.asarray(column_data.values, dtype=float)
            n = x.size
            n_blocks = n // w
            if n_blocks < 1:
                return pd.Series([], dtype=float)
            x2 = x[: n_blocks * w].reshape(n_blocks, w)
            block_means = x2.mean(axis=1)
            if time_values is not None:
                t = time_values[: n_blocks * w].reshape(n_blocks, w)
                idx = t.mean(axis=1)
            else:
                idx = (np.arange(n_blocks) * w) + (w // 2)
            return pd.Series(block_means, index=idx)

        else:
            raise ValueError("Invalid method. Choose 'sliding' or 'non-overlapping'.")

    def _time_values_for_series(self, series: pd.Series) -> Optional[np.ndarray]:
        """Return the 'time' column values aligned to `series.index`, or None."""
        if "time" not in self.data.columns:
            return None
        try:
            return self.data.loc[series.index, "time"].to_numpy(dtype=float)
        except Exception:
            return None

    # ----------- Window autotune (tau_int + Ljung-Box) ----------------

    def _estimate_tau_int(self, series: pd.Series) -> float:
        """Estimate tau_int from raw series ACF using Geyer positive-pair truncation."""
        x = np.asarray(series.dropna().values, dtype=float)
        n = x.size
        if n < 3:
            return 1.0
        nlags = max(1, min(n // 4, 2000))
        r = acf(x, nlags=nlags, fft=False)
        return _tau_int_geyer_from_acf(r)

    def _autotune_window_size(
        self,
        col: str,
        series: pd.Series,
        method: str = "non-overlapping",
        alpha: float = 0.05,
        lag_set=(5, 10),
        B_min: int = 15,
        c0: float = 2.0,
        max_iter: int = 20,
        w_min: int = 5,
    ):
        """
        Choose block window size via:
          1) tau_int estimate (Geyer on raw ACF)
          2) w0 = ceil(c0 * tau_int)
          3) Ljung-Box test on block means; grow w by +1 on failure
          4) enforce B_min blocks where possible
          5) fallback to best p-value if independence never achieved

        Returns
        -------
        (chosen_w: int, info: dict)
        """
        x = series.dropna()
        time_values = self._time_values_for_series(x)
        n = int(x.size)
        if n < 2:
            return w_min, {"status": "too_few_blocks", "tau_int": 1.0, "chosen_w": w_min, "passed": False}

        tau = self._estimate_tau_int(x)
        w = int(max(w_min, math.ceil(c0 * tau)))

        # Cap to keep >= B_min blocks where possible
        w_cap = (n // B_min) if (B_min and n // B_min >= 1) else n
        cap_applied = w_cap >= w_min and w > w_cap
        if cap_applied:
            w = min(w, w_cap)

        best = {"w": w, "passed": False, "p_min": -np.inf, "lb": {"lags": [], "pvalues": []}}

        for _ in range(int(max_iter)):
            proc = self._process_column(x, w, method, time_values=time_values)
            bm = np.asarray(proc.values, dtype=float)
            if bm.size < 2:
                break
            passed, det = _ljung_box_pass(bm, alpha=alpha, lag_set=lag_set)
            p_score = min(det["pvalues"]) if det.get("pvalues") else -np.inf
            if p_score > best["p_min"]:
                best = {"w": int(w), "passed": bool(passed), "p_min": float(p_score), "lb": det}
            if passed:
                return int(w), {"status": "independent", "tau_int": float(tau), "chosen_w": int(w), "passed": True, "lb": det}
            w_next = w + 1
            if w_next <= w or n // w_next < 2:
                break
            w = w_next

        if best["p_min"] > -np.inf:
            return int(best["w"]), {"status": "best_p", "tau_int": float(tau), "chosen_w": int(best["w"]), "passed": False, "lb": best["lb"]}

        return int(w), {"status": "too_few_blocks", "tau_int": float(tau), "chosen_w": int(w), "passed": False, "lb": {"lags": [], "pvalues": []}}

    # ----------- Block-level ESS + variance ----------------

    def get_block_effective_n(
        self,
        column_name: str,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
    ) -> dict:
        """
        Compute Geyer ESS on block means (not on raw series).

        Returns
        -------
        dict: {"effective_n": float, "window_size": int, "n_blocks": int}
        """
        series = self.data[column_name].dropna()
        if series.empty:
            return {"effective_n": float("nan"), "window_size": window_size or 0, "n_blocks": 0}
        est_win = self._estimate_window(column_name, series, window_size)
        time_values = self._time_values_for_series(series)
        proc = self._process_column(series, est_win, method, time_values=time_values)
        bm = proc.values
        n_blocks = len(bm)
        ess_blocks = _geyer_ess_on_blocks(bm) if n_blocks >= 3 else 1.0
        return {"effective_n": float(ess_blocks), "window_size": int(est_win), "n_blocks": int(n_blocks)}

    def _variance(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ) -> dict:
        """Variance of block means, using same window as _mean_uncertainty."""
        results = {}
        for col in self._get_columns(column_name):
            series = self.data[col].dropna()
            if series.empty:
                results[col] = {"variance": np.nan, "window_size": np.nan, "effective_n_blocks": np.nan}
                continue
            info = self.get_block_effective_n(col, method=method, window_size=window_size)
            est_win = info["window_size"]
            time_values = self._time_values_for_series(series)
            proc = self._process_column(series, est_win, method, time_values=time_values)
            bm = proc.values
            var_val = float(np.var(bm, ddof=ddof)) if bm.size >= 2 else np.nan
            results[col] = {"variance": var_val, "window_size": int(est_win), "effective_n_blocks": info["effective_n"]}
        return results

    def _short_term_counts(
        self,
        column_name=None,
        method="non-overlapping",
        window_size=None,
    ) -> dict:
        """Count of non-overlapping block means used in block statistics."""
        results = {}
        for col in self._get_columns(column_name):
            series = self.data[col].dropna()
            if series.empty:
                results[col] = {"n_short_averages": 0, "window_size": 0, "n_blocks": 0}
                continue
            est_win = self._estimate_window(col, series, window_size)
            if len(series) < est_win:
                results[col] = {"n_short_averages": 0, "window_size": int(est_win), "n_blocks": 0}
                continue
            time_values = self._time_values_for_series(series)
            proc = self._process_column(series, est_win, method, time_values=time_values)
            n = int(len(proc))
            results[col] = {"n_short_averages": n, "window_size": int(est_win), "n_blocks": n}
        return results

    # ----------- ESS (classic and robust) ----------------
    def effective_sample_size(self, column_names=None, alpha=0.05):
        """
        Compute classic ESS based on significant autocorrelation lags.

        Parameters
        ----------
        column_names : str or list of str or None
            Columns to compute ESS for; defaults to all except 'time'.
        alpha : float
            Significance level for autocorrelation cutoff.

        Returns
        -------
        dict
            {'results': {col: ESS_int or message}}
        """

        columns = _resolve_columns(self.data, column_names)
        results = {col: _compute_ess(self.data, col, alpha) for col in columns}

        return {
            "results": to_native_types(results),
        }

    @staticmethod
    def robust_effective_sample_size(
        x,
        rank_normalize=True,
        min_samples=8,
        return_relative=False,
    ):
        """
        Compute a robust ESS via pairwise autocorrelations and optional rank-normalization.

        Parameters
        ----------
        x : array-like
        rank_normalize : bool
        min_samples : int
        return_relative : bool

        Returns
        -------
        float or tuple
            ESS (and ESS/n ratio if return_relative).
        """
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        n = len(x)
        if n < min_samples:
            return np.nan if not return_relative else (np.nan, np.nan)
        if np.all(x == x[0]):
            return float(n) if not return_relative else (float(n), 1.0)
        if rank_normalize:
            x = rankdata(x, method="average")
            x = (x - np.mean(x)) / np.std(x, ddof=0)
        else:
            x = x - np.mean(x)
        var = np.var(x, ddof=0)
        if var == 0:
            return float(n) if not return_relative else (float(n), 1.0)
        acorr = np.empty(n)
        acorr[0] = 1.0
        for lag in range(1, n):
            v1 = x[:-lag]
            v2 = x[lag:]
            acorr[lag] = np.dot(v1, v2) / ((n - lag) * var)
        s = 0.0
        t = 1
        while t + 1 < n:
            pair_sum = acorr[t] + acorr[t + 1]
            if pair_sum < 0:
                break
            s += pair_sum
            t += 2
        ess = n / (1 + 2 * s)
        ess = max(1.0, min(ess, n))
        if return_relative:
            return ess, ess / n
        return ess

    def ess_robust(
        self,
        column_names=None,
        rank_normalize=False,
        min_samples=8,
        return_relative=False,
    ):
        """
        Wrapper for `robust_effective_sample_size` over multiple columns.

        Records the operation in history.

        Parameters
        ----------
        column_names : str or list or None
        rank_normalize : bool
        min_samples : int
        return_relative : bool

        Returns
        -------
        dict
            {'results': {col: ESS or tuple}}
        """

        if column_names is None:
            column_names = [col for col in self.data.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
        results = {}
        for col in column_names:
            if col not in self.data.columns:
                results[col] = {"error": f"Column '{col}' not found."}
                continue
            x = self.data[col].dropna().values
            ess = self.robust_effective_sample_size(
                x,
                rank_normalize=rank_normalize,
                min_samples=min_samples,
                return_relative=return_relative,
            )
            results[col] = ess

        return {"results": to_native_types(results)}

    @staticmethod
    def normalize_data(df):
        """
        Min-Max normalize all signal columns (excluding 'time') to [0,1].

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        scaler = MinMaxScaler()
        signal_columns = df.columns[1:]
        df[signal_columns] = df[signal_columns].astype(float)
        df[signal_columns] = scaler.fit_transform(df[signal_columns])
        return df
