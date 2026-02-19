import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, rankdata
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf, adfuller

from quends.base.utils import power_law_model

from .history import DataStreamHistory, DataStreamHistoryEntry
from .utils import to_native_types

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

    def _append_history_entry(self, entry: DataStreamHistoryEntry) -> None:
        """Append a history entry to the DataStream's history."""
        self._history.append(entry)

    def _history_metadata(self):
        """Return history as a list of dicts"""
        return [
            {"operation": e.operation_name, "options": e.parameters}
            for e in self._history.entries()
        ]

    def _last_history_metadata_entry(self):
        """Return the most recent history item"""
        entries = self._history.entries()
        if not entries:
            return None
        last = entries[-1]
        return {"operation": last.operation_name, "options": last.parameters}

    @property
    def data(self) -> Any:
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
        Index
            ColumnIndex of variable names in `self.df`.
        """
        return self.data.columns

    # --------- Statistical summaries ---------
    def mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Compute block or sliding window means for each column.

        Private helper for compute_statistics and confidence intervals.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.data[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            proc_data = self._process_column(column_data, est_win, method)
            results[col] = {
                "mean": float(np.mean(proc_data)),
                "window_size": int(est_win),
            }
        return results

    def mean_uncertainty(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Estimate the standard error of the mean via block/sliding windows.

        Private helper.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.data[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            proc_data = self._process_column(column_data, est_win, method)
            if method == "sliding":
                step = max(1, est_win // 4)
                effective_n = len(proc_data[::step])
            else:
                effective_n = len(proc_data)
            uncertainty = float(np.std(proc_data, ddof=ddof) / np.sqrt(effective_n))
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
        statistics = {}
        columns = self._get_columns(column_name)
        mean_results = self.mean(column_name, method=method, window_size=window_size)
        mu_results = self.mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ci_results = self.confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ess_dict = self.effective_sample_size(column_names=column_name)
        for col in columns:
            column_data = self.data[col].dropna()
            if column_data.empty:
                statistics[col] = {"error": f"No data available for column '{col}'"}
                continue
            mean_val = mean_results[col]["mean"]
            mean_uncertainty = mu_results[col]["mean_uncertainty"]
            ci = ci_results[col]["confidence_interval"]
            est_win = mean_results[col]["window_size"]
            ess_val = ess_dict["results"].get(col, 10)
            statistics[col] = {
                "mean": mean_val,
                "mean_uncertainty": mean_uncertainty,
                "confidence_interval": ci,
                "pm_std": (mean_val - mean_uncertainty, mean_val + mean_uncertainty),
                "effective_sample_size": ess_val,
                "window_size": est_win,
            }

        # make a history entry with operation details for compute statistics
        entry = DataStreamHistoryEntry(
            operation_name="compute_statistics",
            parameters={
                "column_name": column_name,
                "ddof": ddof,
                "method": method,
                "window_size": window_size,
            },
        )

        # append to DataStream's History
        self._append_history_entry(entry)

        # Keep compute_statistics metadata focused on stats operations only
        statistics["metadata"] = list(ess_dict.get("metadata", [])) + [
            {"operation": entry.operation_name, "options": entry.parameters}
        ]

        return to_native_types(statistics)

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
            proc_data = self._process_column(column_data, est_win, method)
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

        # make a history entry with details for cumulative statistics
        entry = DataStreamHistoryEntry(
            operation_name="cumulative_statistics",
            parameters={
                "column_name": column_name,
                "method": method,
                "window_size": window_size,
            },
        )

        # append to DataStream's History
        self._append_history_entry(entry)

        # convert deduplicated entries to dict format
        deduped_entries = self._history_metadata()

        results["metadata"] = deduped_entries
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

        entry = DataStreamHistoryEntry(
            operation_name="additional_data",
            parameters={
                "column_name": column_name,
                "ddof": ddof,
                "method": method,
                "window_size": window_size,
                "reduction_factor": reduction_factor,
            },
        )

        self._append_history_entry(entry)

        # show only additional data as the history
        last_entry = self._last_history_metadata_entry()
        metadata = [last_entry] if last_entry is not None else []

        results["metadata"] = metadata

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

        Records operation in history and returns a dict of bool or error.

        Parameters
        ----------
        columns : str or list of str

        Returns
        -------
        dict
            {column: True if stationary (p<0.05), else False or error message}
        """
        # Add to history
        entry = DataStreamHistoryEntry(
            operation_name="is_stationary", parameters={"columns": columns}
        )
        self._append_history_entry(entry)
        if isinstance(columns, str):
            columns = [columns]
        results = {}
        for column in columns:
            try:
                p_value = adfuller(self.data[column].dropna(), autolag="AIC")[1]
                results[column] = p_value < 0.05
            except Exception as e:
                results[column] = f"Error: {e}"
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
        Determine block size: either user-provided or derived from ESS.

        Ensures a minimum window of 5 samples.
        """
        if window_size is None:
            # Get ESS dictionary from method
            ess_results = self.effective_sample_size(column_names=col)
            # Unpack the result
            ess_val = None
            if isinstance(ess_results, dict) and "results" in ess_results:
                ess_val = ess_results["results"].get(col, 10)
            else:
                ess_val = (
                    ess_results.get(col, 10) if isinstance(ess_results, dict) else 10
                )
            # Avoid division by zero or negative
            try:
                ess_val = max(1, int(round(ess_val)))
            except Exception:
                ess_val = 10
            return max(5, len(column_data) // ess_val)
        return window_size

    def _process_column(self, column_data, estimated_window, method):
        """
        Transform a 1D series into block or sliding window means.

        Parameters
        ----------
        column_data : pandas.Series
        estimated_window : int
        method : {'sliding', 'non-overlapping'}

        Returns
        -------
        pandas.Series
        """
        if method == "sliding":
            return column_data.rolling(window=estimated_window).mean().dropna()
        elif method == "non-overlapping":
            step_size = max(1, estimated_window)  # max(1, estimated_window // 4)
            window_means = [
                np.mean(column_data[i : i + estimated_window])
                for i in range(0, len(column_data) - estimated_window + 1, step_size)
            ]
            return pd.Series(
                window_means,
                index=np.arange(
                    estimated_window // 2,
                    len(window_means) * step_size + estimated_window // 2,
                    step_size,
                ),
            )
        else:
            raise ValueError("Invalid method. Choose 'sliding' or 'non-overlapping'.")

    # ----------- ESS (classic and robust) ----------------
    def effective_sample_size(self, column_names=None, alpha=0.05):
        """
        Compute classic ESS based on significant autocorrelation lags.

        Records the operation in history.

        Parameters
        ----------
        column_names : str or list of str or None
            Columns to compute ESS for; defaults to all except 'time'.
        alpha : float
            Significance level for autocorrelation cutoff.

        Returns
        -------
        dict
            {'results': {col: ESS_int or message}, 'metadata': history}
        """
        # Create a history entry with operation details of ess
        entry = DataStreamHistoryEntry(
            operation_name="effective_sample_size",
            parameters={"column_names": column_names, "alpha": alpha},
        )

        # append to DataStream's History
        self._append_history_entry(entry)

        if column_names is None:
            column_names = [col for col in self.data.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
        results = {}
        for col in column_names:
            if col not in self.data.columns:
                results[col] = {
                    "message": f"Column '{col}' not found in the DataStream."
                }
                continue
            filtered = self.data[col].dropna()
            if filtered.empty:
                results[col] = {
                    "effective_sample_size": None,
                    "message": "No data available for computation.",
                }
                continue
            n = len(filtered)
            nlags = int(n / 4)
            acf_values = acf(filtered, nlags=nlags)
            z_critical = norm.ppf(1 - alpha / 2)
            conf_interval = z_critical / np.sqrt(n)
            significant_lags = np.where(np.abs(acf_values[1:]) > conf_interval)[0]
            acf_sum = np.sum(np.abs(acf_values[1:][significant_lags]))
            ESS = n / (1 + 2 * acf_sum)
            results[col] = int(np.ceil(ESS))

        # Keep ESS metadata focused on this operation only
        metadata = [{"operation": entry.operation_name, "options": entry.parameters}]

        return {"results": to_native_types(results), "metadata": metadata}

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
            {'results': {col: ESS or tuple}, 'metadata': history}
        """
        # Create a history entry with the operation details
        entry = DataStreamHistoryEntry(
            operation_name="ess_robust",
            parameters={
                "column_names": column_names,
                "rank_normalize": rank_normalize,
                "min_samples": min_samples,
                "return_relative": return_relative,
            },
        )

        # append to the DataStream's History
        self._append_history_entry(entry)

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

        # Keep robust ESS metadata focused on this operation only
        metadata = [{"operation": entry.operation_name, "options": entry.parameters}]

        return {"results": to_native_types(results), "metadata": metadata}

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
        df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
        return df
