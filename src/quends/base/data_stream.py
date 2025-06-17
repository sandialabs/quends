"""
data_stream.py

Enhanced DataStream for robust scientific reproducibility.

- Tracks all options and arguments for every operation (trim, statistics, etc.).
- Each result dict includes full lineage of all processing steps and their options.
- Always auto-skips to first nonzero entry before steady-state detection in trim.
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.robust.scale import mad
from statsmodels.tsa.stattools import acf, adfuller

class DataStream:
    """
    Scientific time-series and simulation trace analysis with full pipeline reproducibility.

    Each DataStream remembers the full chain of analysis steps and their options
    (in self._history), and every result returned includes this lineage.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with a 'time' column and one or more data columns.
    _history : list, optional
        Internal: Lineage of operations (not user-supplied).
    """

    def __init__(self, df: pd.DataFrame, _history=None):
        self.df = df
        self._history = list(_history) if _history is not None else []

    def _add_history(self, operation, options):
        """Append an operation and its options to this instance's history."""
        options = {k: v for k, v in options.items() if k not in ('self', 'cls', '__class__')}
        self._history.append({
            "operation": operation,
            "options": options
        })

    def head(self, n=5):
        """Return the first n rows of the underlying DataFrame."""
        return self.df.head(n)

    def __len__(self):
        """Return the number of rows in the DataStream."""
        return len(self.df)

    def variables(self):
        """List the variable (column) names in the DataStream."""
        return self.df.columns

    # ---------- Main Trim and Detection Logic ----------

    def trim(
        self,
        column_name,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
    ):
        """
        Trim the DataStream to start from steady-state using a chosen method.
        Handles all start time logic (user-specified and auto-skip to first nonzero).
        Tracks options for reproducibility.

        Parameters
        ----------
        column_name : str
            Name of the column to analyze.
        batch_size : int, optional
            Rolling/statistical window size for steady-state detection (default: 10).
        start_time : float, optional
            User-specified start time for analysis (default: 0.0).
        method : {"std", "threshold", "rolling_variance"}, optional
            Method for steady-state detection.
        threshold : float or None, optional
            Threshold for steady-state (used in "threshold"/"rolling_variance").
        robust : bool, optional
            Use robust median/MAD (for "std" method).

        Returns
        -------
        DataStream
            New DataStream with trimmed data, from steady-state onward.

        Raises
        ------
        ValueError
            If the data column is not stationary, or if no steady-state found.
        """
        stationary_result = self.is_stationary(column_name)
        is_stat = stationary_result.get(column_name, False) if isinstance(stationary_result, dict) else bool(stationary_result)
        if not is_stat:
            raise ValueError(
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data. "
                "Apply detrending or differencing first."
            )

        data = self.df[self.df["time"] >= start_time].reset_index(drop=True)
        # Always skip to first nonzero in the target column
        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
            data = data.loc[non_zero_index:].reset_index(drop=True)

        if method == "std":
            steady_state_start_time = self.find_steady_state_std(
                data, column_name, window_size=batch_size, robust=robust
            )
        elif method == "threshold":
            if threshold is None:
                raise ValueError("Threshold must be specified for the 'threshold' method.")
            steady_state_start_time = self.find_steady_state_threshold(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        elif method == "rolling_variance":
            threshold = threshold if threshold is not None else 0.1
            steady_state_start_time = self.find_steady_state_rolling_variance(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        else:
            raise ValueError(
                "Invalid method. Choose 'std', 'threshold', or 'rolling_variance'."
            )

        if steady_state_start_time is not None:
            trimmed_df = self.df.loc[
                self.df["time"] >= steady_state_start_time, ["time", column_name]
            ].reset_index(drop=True)
            new_history = self._history.copy()
            options = dict(
                column_name=column_name, batch_size=batch_size, start_time=start_time,
                method=method, threshold=threshold, robust=robust,
                operation_detected_time=steady_state_start_time,
            )
            new_history.append({"operation": "trim", "options": options})
            return DataStream(trimmed_df, _history=new_history)
        else:
            raise ValueError(
                f"Steady-state start time could not be determined for column '{column_name}'."
            )

    @staticmethod
    def find_steady_state_std(
        data, column_name, window_size=10, robust=True
    ):
        """
        Find the steady-state start time based on std or robust median/MAD method.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with the time series (already filtered).
        column_name : str
            Column to analyze.
        window_size : int, optional
            Window size (default: 10).
        robust : bool, optional
            Use median/MAD if True, else mean/std (default: True).

        Returns
        -------
        float or None
            Detected steady-state start time, or None if not found.
        """
        time_filtered = data["time"].values
        signal_filtered = data[column_name].values

        for i in range(len(signal_filtered) - window_size + 1):
            remaining_data = signal_filtered[i:]
            if robust:
                central_value = np.median(remaining_data)
                scale_value = mad(remaining_data)
            else:
                central_value = np.mean(remaining_data)
                scale_value = np.std(remaining_data)

            within_1 = np.mean(np.abs(remaining_data - central_value) <= scale_value)
            within_2 = np.mean(np.abs(remaining_data - central_value) <= 2 * scale_value)
            within_3 = np.mean(np.abs(remaining_data - central_value) <= 3 * scale_value)

            if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
                return time_filtered[i]
        return None

    @staticmethod
    def find_steady_state_rolling_variance(
        data, column_name, window_size=50, threshold=0.1
    ):
        """
        Find the steady-state start time based on rolling variance.

        Parameters
        ----------
        data : pandas.DataFrame
            Time series data (already filtered).
        column_name : str
            Column to analyze.
        window_size : int, optional
            Window size for rolling variance (default: 50).
        threshold : float, optional
            Relative threshold for variance (default: 0.1).

        Returns
        -------
        float or None
            Detected steady-state start time, or None if not found.
        """
        ts = data[["time", column_name]].dropna()
        time_values = ts["time"]
        signal_values = ts[column_name]

        rolling_variance = signal_values.rolling(window=window_size).var()
        threshold_val = rolling_variance.mean() * threshold
        steady_state_index = np.where(rolling_variance < threshold_val)[0]

        if len(steady_state_index) > 0:
            return time_values.iloc[steady_state_index[0]]
        return None

    @staticmethod
    def normalize_data(df):
        """
        Normalize data columns (excluding 'time') to [0, 1] for robust analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to normalize.

        Returns
        -------
        pandas.DataFrame
            Normalized DataFrame.
        """
        scaler = MinMaxScaler()
        df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
        return df

    @staticmethod
    def find_steady_state_threshold(
        data, column_name, window_size, threshold
    ):
        """
        Detect steady-state start time based on rolling std below a fixed threshold.

        Parameters
        ----------
        data : pandas.DataFrame
            Input time series (already filtered).
        column_name : str
            Data column to analyze.
        window_size : int
            Rolling window size.
        threshold : float
            Std threshold for detection.

        Returns
        -------
        float or None
            Detected steady-state time, or None if not found.
        """
        normalized_data = DataStream.normalize_data(data.copy())
        time_series = normalized_data[["time", column_name]]

        if len(time_series) < window_size:
            return None

        rolling_std = time_series[column_name].rolling(window=window_size).std().rolling(
            window=window_size
        ).std() / np.sqrt(window_size)
        common_idx = time_series.index.intersection(rolling_std.index)
        steady_state = time_series.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]
        if not steady_state.empty:
            return steady_state.iloc[0]
        return None

    # ============= Statistical Methods with Option Tracking ==============

    def _mean(self, column_name=None, method="non-overlapping", window_size=None):
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            proc_data = self._process_column(column_data, est_win, method)
            results[col] = {"mean": np.mean(proc_data), "window_size": est_win}
        return results

    def _mean_uncertainty(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
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
            uncertainty = np.std(proc_data, ddof=ddof) / np.sqrt(effective_n)
            results[col] = {"mean_uncertainty": uncertainty, "window_size": est_win}
        return results

    def _confidence_interval(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        results = {}
        mean_results = self._mean(column_name, method=method, window_size=window_size)
        uncertainty_results = self._mean_uncertainty(
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
                "confidence_interval": (ci_lower, ci_upper),
                "window_size": mean_results[col]["window_size"],
            }
        return results

    def compute_statistics(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Calculate summary statistics for each column:
        mean, uncertainty, confidence interval, ±std, effective sample size (ESS), and window_size used.
        Returns the full operation lineage in "_history".

        Parameters
        ----------
        column_name : str or list, optional
            Columns to compute statistics for. If None, infer columns.
        ddof : int, optional
            Delta degrees of freedom for std.
        method : {"sliding", "non-overlapping"}, optional
            How to calculate windowed means.
        window_size : int, optional
            Window/batch size for the selected method. If None, estimated.

        Returns
        -------
        dict
            For each column:
              - mean
              - mean_uncertainty
              - confidence_interval
              - pm_std
              - effective_sample_size
              - window_size
            Plus:
              - "_history" (list of all options for all steps)
        """
        statistics = {}
        columns = self._get_columns(column_name)

        mean_results = self._mean(column_name, method=method, window_size=window_size)
        mu_results = self._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ci_results = self._confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )

        for col in columns:
            column_data = self.df[col].dropna()
            if column_data.empty:
                statistics[col] = {"error": f"No data available for column '{col}'"}
                continue
            ess_dict = self.effective_sample_size(column_names=col)
            ess_val = ess_dict.get(col, 10)  # fallback if not available

            mean_val = mean_results[col]["mean"]
            mean_uncertainty = mu_results[col]["mean_uncertainty"]
            ci = ci_results[col]["confidence_interval"]
            est_win = mean_results[col]["window_size"]

            statistics[col] = {
                "mean": mean_val,
                "mean_uncertainty": mean_uncertainty,
                "confidence_interval": ci,
                "pm_std": (mean_val - mean_uncertainty, mean_val + mean_uncertainty),
                "effective_sample_size": ess_val,
                "window_size": est_win,
            }

        op_options = dict(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size,
        )
        full_history = self._history.copy()
        full_history.append({"operation": "compute_statistics", "options": op_options})
        statistics["_history"] = full_history
        return statistics

    # ----------- Helper methods for modularity (unchanged except for window_size name) -----------

    def _get_columns(self, column_name):
        if column_name is None:
            return [col for col in self.df.columns if col != "time"]
        return [column_name] if isinstance(column_name, str) else column_name

    def _estimate_window(self, col, column_data, window_size):
        if window_size is None:
            ess_results = self.effective_sample_size(column_names=col)
            ess_value = ess_results.get(col, 10)
            return max(5, len(column_data) // ess_value)
        return window_size

    def _process_column(self, column_data, estimated_window, method):
        if method == "sliding":
            return column_data.rolling(window=estimated_window).mean().dropna()
        elif method == "non-overlapping":
            step_size = max(1, estimated_window // 4)
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

    # ------- Other methods (stationarity, ESS, cumulative, etc.) can also use _add_history -------

    def is_stationary(self, columns):
        """
        Check if specified columns are stationary using the ADF test.
        Tracks options used for reproducibility.

        Parameters
        ----------
        columns : str or list
            Column(s) to check for stationarity.

        Returns
        -------
        dict
            {column_name: bool or error string}
        """
        # Add to history for this operation
        self._add_history("is_stationary", {"columns": columns})

        if isinstance(columns, str):
            columns = [columns]

        results = {}
        for column in columns:
            try:
                p_value = adfuller(self.df[column].dropna(), autolag="AIC")[1]
                results[column] = p_value < 0.05
            except ValueError as e:
                results[column] = f"Error: {e}"

        return results

    def effective_sample_size(self, column_names=None, alpha=0.05):
        """
        Compute the effective sample size (ESS) for the data using autocorrelation.

        Parameters
        ----------
        column_names : str or list, optional
            Columns to compute ESS for (default: all except 'time').
        alpha : float, optional
            Significance level for confidence interval (default 0.05).

        Returns
        -------
        dict
            {column_name: ESS}
        """
        self._add_history("effective_sample_size", {"column_names": column_names, "alpha": alpha})

        if column_names is None:
            column_names = [col for col in self.df.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]

        results = {}
        for col in column_names:
            if col not in self.df.columns:
                results[col] = {
                    "message": f"Column '{col}' not found in the DataStream."
                }
                continue
            filtered = self.df[col].dropna()
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
        return results

    def cumulative_statistics(
            self, column_name=None, method="non-overlapping", window_size=None
            ):
        """
        Compute cumulative statistics: mean, std, and SEM (std error of mean) as function of sample size.

        Parameters
        ----------
        column_name : str, list, or None, optional
            Columns to process.
        method : {"sliding", "non-overlapping"}, optional
            Window method.
        window_size : int or None, optional
            Window size.

        Returns
        -------
        dict
            {column: {"cumulative_mean": [...], "cumulative_uncertainty": [...], "standard_error": [...]} }
            Plus: '_history' key showing full options chain.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
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
                "window_size": est_win
        }

        op_options = dict(
            column_name=column_name, method=method, window_size=window_size
        )
        full_history = self._history.copy()
        full_history.append({"operation": "cumulative_statistics", "options": op_options})
        results["_history"] = full_history
        return results


    def additional_data(
        self,
        column_name=None,
        ddof=1,
        method="sliding",
        window_size=None,
        reduction_factor=0.1,
    ):
        """
        Fit a power-law SEM(n) = A/n^p to cumulative SEM, and estimate extra samples
        needed to reduce SEM by a given factor.

        Parameters
        ----------
        column_name : str, list, or None, optional
            Columns to process.
        ddof : int, optional
            Degrees of freedom for std.
        method : {"sliding", "non-overlapping"}, optional
            Windowing method.
        window_size : int or None, optional
            Window size.
        reduction_factor : float, optional
            Fractional SEM reduction target (e.g. 0.1 for 10%).

        Returns
        -------
        dict
            {column: {...fit results, n_target, additional_samples...}, "_history": full history}
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

            column_data = self.df[col].dropna()
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

            def power_law_model(n, A, p):
                return A / (n**p)

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
                "A_est": A_est,
                "p_est": p_est,
                "n_current": n_current,
                "current_sem": current_sem,
                "target_sem": target_sem,
                "n_target": n_target,
                "additional_samples": math.ceil(additional_samples),
                "window_size": est_win
            }

        op_options = dict(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size, reduction_factor=reduction_factor
        )
        full_history = self._history.copy()
        full_history.append({"operation": "additional_data", "options": op_options})
        results["_history"] = full_history
        return results


    def effective_sample_size_below(self, column_names=None, alpha=0.05):
        """
        Estimate ESS using all lags up to where |ACF| drops below confidence bound.

        Parameters
        ----------
        column_names : str, list, or None, optional
            Columns to analyze.
        alpha : float, optional
            Confidence interval significance level (default: 0.05).

        Returns
        -------
        dict
            {column: ESS, ... "_history": full history }
        """
        self._add_history("effective_sample_size_below", {"column_names": column_names, "alpha": alpha})

        if column_names is None:
            column_names = [col for col in self.df.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]

        results = {}
        for col in column_names:
            if col not in self.df.columns:
                results[col] = {
                    "message": f"Column '{col}' not found in the DataStream."
                }
                continue

            filtered = self.df[col].dropna()
            if filtered.empty:
                results[col] = {
                    "effective_sample_size": None,
                    "message": "No data available for computation.",
                }
                continue

            n = len(filtered)
            nlags = int(n / 3)
            acf_values = acf(filtered, nlags=nlags, fft=False)
            z_critical = norm.ppf(1 - alpha / 2)
            conf_interval = z_critical / np.sqrt(n)

            significant_idx = None
            for i in range(1, len(acf_values)):
                if np.abs(acf_values[i]) < conf_interval:
                    significant_idx = i
                    break
            if significant_idx is None:
                significant_lags = np.arange(1, len(acf_values))
            else:
                significant_lags = np.arange(1, significant_idx)

            acf_sum = np.sum(np.abs(acf_values[significant_lags]))
            ESS = n / (1 + 2 * acf_sum)
            results[col] = int(np.floor(ESS))

        results["_history"] = self._history.copy()
        return results

