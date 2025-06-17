"""
data_stream.py

Provides the DataStream class for statistical analysis, steady-state detection, stationarity testing,
and uncertainty quantification on time series data (as pandas DataFrames). Designed for scientific
simulation outputs and ensemble data workflows.

Author: [Your Name]
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
    A wrapper for time-series data (pandas DataFrame) that provides rich statistical and
    steady-state analysis for simulation and experimental outputs.

    This class enables:
      - Steady-state trimming (using std, threshold, or rolling-variance methods)
      - Stationarity testing (ADF)
      - Effective sample size calculation
      - Mean, uncertainty, and confidence interval computation (with windowing)
      - Cumulative statistics and power-law SEM extrapolation

    Parameters
    ----------
    df : pandas.DataFrame
        The time series data. Must have a 'time' column and one or more data columns.

    Examples
    --------
    >>> ds = DataStream(my_dataframe)
    >>> ds.mean('flux', method='sliding', window_size=20)
    >>> trimmed = ds.trim('flux', method='std')
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize a DataStream object.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing 'time' and data columns.
        """
        self.df = df

    def head(self, n=5):
        """
        Return the first n rows of the underlying DataFrame.

        Parameters
        ----------
        n : int, optional
            Number of rows to return (default: 5).

        Returns
        -------
        pandas.DataFrame
            The first n rows.
        """
        return self.df.head(n)

    def __len__(self):
        """
        Return the length (number of rows) of the DataStream.

        Returns
        -------
        int
            Number of data points (rows).
        """
        return len(self.df)

    def variables(self):
        """
        List the variable (column) names in the DataStream.

        Returns
        -------
        pandas.Index
            Names of the columns in the DataFrame.
        """
        return self.df.columns

    # --- Internal Helper Methods ---

    def _get_columns(self, column_name):
        """
        Infer a list of columns to operate on based on user input.

        Parameters
        ----------
        column_name : str, list, or None
            Single column name, list of names, or None (use all except 'time').

        Returns
        -------
        list of str
            The column names to process.
        """
        if column_name is None:
            return [col for col in self.df.columns if col != "time"]
        return [column_name] if isinstance(column_name, str) else column_name

    def _estimate_window(self, col, column_data, window_size):
        """
        Estimate a window size for rolling/statistics based on ESS or user input.

        Parameters
        ----------
        col : str
            Column name.
        column_data : pandas.Series
            Data to analyze.
        window_size : int or None
            User-provided window, or None to estimate.

        Returns
        -------
        int
            Window size to use.
        """
        if window_size is None:
            ess_results = self.effective_sample_size(column_names=col)
            ess_value = ess_results.get(col, 10)
            return max(5, len(column_data) // ess_value)
        return window_size

    def _process_column(self, column_data, estimated_window, method):
        """
        Apply windowed averaging (sliding or non-overlapping) to a column.

        Parameters
        ----------
        column_data : pandas.Series
            Data to process.
        estimated_window : int
            Window size to use.
        method : {"sliding", "non-overlapping"}
            Windowing approach.

        Returns
        -------
        pandas.Series
            Series of window means.
        """
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

    # =======================
    #    Main Functionality
    # =======================

    def trim(
        self,
        column_name,
        window_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
    ):
        """
        Trim the DataStream to start from steady-state using a chosen method.

        Parameters
        ----------
        column_name : str
            Name of the column to analyze.
        window_size : int, optional
            Window size for rolling/statistics (default: 10).
        start_time : float, optional
            Start time for the analysis (default: 0.0).
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
        """
        if method == "std":
            steady_state_start_time = self.find_steady_state_std(
                self.df, column_name, window_size, start_time, robust
            )
        elif method == "threshold":
            if threshold is None:
                raise ValueError(
                    "Threshold must be specified for the 'threshold' method."
                )
            steady_state_start_time = self.find_steady_state_threshold(
                self.df, column_name, window_size, threshold, start_time
            )
        elif method == "rolling_variance":
            threshold = threshold if threshold is not None else 0.1
            steady_state_start_time = self.find_steady_state_rolling_variance(
                self.df, column_name, window_size, threshold
            )
        else:
            raise ValueError(
                "Invalid method. Choose 'std', 'threshold', or 'rolling_variance'."
            )

        if steady_state_start_time is not None:
            trimmed_df = self.df.loc[
                self.df["time"] >= steady_state_start_time, ["time", column_name]
            ].reset_index(drop=True)
            return DataStream(trimmed_df)

    @staticmethod
    def find_steady_state_std(
        data, column_name, window_size=10, start_time=0.0, robust=True
    ):
        """
        Locate the onset of steady-state using std or median/MAD windows.

        This method slides a window and checks if the fraction of values within 1, 2, and 3 std/MAD
        matches the theoretical normal-distribution probabilities (0.68, 0.95, 0.99).

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with the time series.
        column_name : str
            Column to analyze.
        window_size : int, optional
            Window size (default: 10).
        start_time : float, optional
            Start time (default: 0.0).
        robust : bool, optional
            Use median/MAD if True, else mean/std (default: True).

        Returns
        -------
        float or None
            Detected steady-state start time, or None if not found.
        """
        if start_time == 0.0:
            non_zero_index = data[data[column_name] > 0].index.min()
            if non_zero_index is not None:
                start_time = data.loc[non_zero_index, "time"]

        filtered_data = data.loc[data["time"] >= start_time]
        time_filtered = filtered_data["time"].values
        signal_filtered = filtered_data[column_name].values

        for i in range(len(signal_filtered) - window_size + 1):
            remaining_data = signal_filtered[i:]
            if robust:
                central_value = np.median(remaining_data)
                scale_value = mad(remaining_data)
            else:
                central_value = np.mean(remaining_data)
                scale_value = np.std(remaining_data)

            within_1 = np.mean(np.abs(remaining_data - central_value) <= scale_value)
            within_2 = np.mean(
                np.abs(remaining_data - central_value) <= 2 * scale_value
            )
            within_3 = np.mean(
                np.abs(remaining_data - central_value) <= 3 * scale_value
            )

            if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
                return time_filtered[i]
        return None

    @staticmethod
    def find_steady_state_rolling_variance(
        data, column_name, window_size=50, threshold=0.1
    ):
        """
        Locate steady-state onset using rolling variance below a threshold.

        Parameters
        ----------
        data : pandas.DataFrame
            Time series data.
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
        data, column_name, window_size, threshold, start_time=0.0
    ):
        """
        Detect steady-state start time based on rolling std below a fixed threshold.

        Parameters
        ----------
        data : pandas.DataFrame
            Input time series.
        column_name : str
            Data column to analyze.
        window_size : int
            Rolling window size.
        threshold : float
            Std threshold for detection.
        start_time : float, optional
            Start time (default: 0.0).

        Returns
        -------
        float or None
            Detected steady-state time, or None if not found.
        """
        if start_time == 0.0:
            non_zero_index = data[data[column_name] > 0].index.min()
            if non_zero_index is not None:
                start_time = data.loc[non_zero_index, "time"]

        normalized_data = DataStream.normalize_data(data.copy())
        time_series = normalized_data[["time", column_name]]
        filtered = time_series.loc[time_series["time"] >= start_time]

        if len(filtered) < window_size:
            return None

        rolling_std = filtered[column_name].rolling(window=window_size).std().rolling(
            window=window_size
        ).std() / np.sqrt(window_size)
        common_idx = filtered.index.intersection(rolling_std.index)
        steady_state = filtered.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]
        if not steady_state.empty:
            return steady_state.iloc[0]
        return None

    def is_stationary(self, columns):
        """
        Test for stationarity using the Augmented Dickey-Fuller (ADF) test.

        Parameters
        ----------
        columns : str or list of str
            Name(s) of columns to test.

        Returns
        -------
        dict
            {column: True/False (stationary), or error message}
        """
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
        Estimate effective sample size (ESS) accounting for autocorrelation.

        Parameters
        ----------
        column_names : str, list, or None, optional
            Columns to analyze (default: all except 'time').
        alpha : float, optional
            Significance level for critical ACF value (default: 0.05).

        Returns
        -------
        dict
            {column: ESS}
        """
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

    def mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Compute mean of (windowed) short-term averages.

        Parameters
        ----------
        column_name : str, list, or None, optional
            Columns to process (default: all except 'time').
        method : {"sliding", "non-overlapping"}, optional
            Type of window averaging (default: "non-overlapping").
        window_size : int or None, optional
            Window size for averaging (default: estimated from ESS).

        Returns
        -------
        dict
            {column: {"mean": float}}
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue

            est_win = self._estimate_window(col, column_data, window_size)
            proc_data = self._process_column(column_data, est_win, method)
            results[col] = {"mean": np.mean(proc_data)}
        return results

    def mean_uncertainty(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Compute uncertainty (standard error) of mean of short-term averages.

        Parameters
        ----------
        column_name : str, list, or None, optional
            Columns to process.
        ddof : int, optional
            Delta degrees of freedom for std (default: 1).
        method : {"sliding", "non-overlapping"}, optional
            Window method (default: "non-overlapping").
        window_size : int or None, optional
            Window size.

        Returns
        -------
        dict
            {column: {"mean uncertainty": float}}
        """
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

            results[col] = {"mean uncertainty": uncertainty}
        return results

    def confidence_interval(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Compute 95% confidence interval for the mean of windowed short-term averages.

        Parameters
        ----------
        column_name : str, list, or None, optional
            Columns to analyze.
        ddof : int, optional
            Delta degrees of freedom for std.
        method : {"sliding", "non-overlapping"}, optional
            Windowing method.
        window_size : int or None, optional
            Window size.

        Returns
        -------
        dict
            {column: {"confidence interval": (lower, upper)}}
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
            uncertainty_val = uncertainty_results[col]["mean uncertainty"]
            ci_lower = mean_val - 1.96 * uncertainty_val
            ci_upper = mean_val + 1.96 * uncertainty_val
            results[col] = {"confidence interval": (ci_lower, ci_upper)}
        return results

    def compute_statistics(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Compute mean, uncertainty, confidence interval, and ±std for columns.

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

        Returns
        -------
        dict
            {column: {...statistics...}}
        """
        mean_results = self.mean(column_name, method=method, window_size=window_size)
        mu_results = self.mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ci_results = self.confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )

        statistics = {}
        for col in mean_results.keys():
            if (
                col not in mean_results
                or col not in mu_results
                or col not in ci_results
            ):
                statistics[col] = {"error": f"Missing data for column '{col}'"}
                continue

            ci = ci_results[col].get("confidence interval")
            if ci is None:
                statistics[col] = {
                    "error": f"Confidence interval not computed for column '{col}'"
                }
                continue

            statistics[col] = {
                "mean": mean_results[col]["mean"],
                "mean_uncertainty": mu_results[col]["mean uncertainty"],
                "confidence_interval": ci,
                "pm_std": (
                    mean_results[col]["mean"] - mu_results[col]["mean uncertainty"],
                    mean_results[col]["mean"] + mu_results[col]["mean uncertainty"],
                ),
            }
        return statistics

    def optimal_window_size(self, column_name=None, method="non-overlapping"):
        """
        Find the window size that minimizes mean uncertainty (std) for a column.

        Parameters
        ----------
        column_name : str, list, or None, optional
            Columns to process.
        method : {"sliding", "non-overlapping"}, optional
            Window method.

        Returns
        -------
        dict
            {column: {...optimal window, std, mean, ci...}}
        """
        if method not in ["sliding", "non-overlapping"]:
            raise ValueError("Invalid method. Choose 'sliding' or 'non-overlapping'.")

        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue

            max_window_size = max(2, len(column_data) // 2)
            std_results = {}
            stats_store = {}

            for window in range(1, max_window_size, 5):
                stats_res = self.compute_statistics(
                    column_name=col, method=method, window_size=window
                )
                std_val = stats_res[col].get("mean_uncertainty")
                mean_val = stats_res[col].get("mean")
                ci_val = stats_res[col].get("confidence_interval")
                if std_val is not None:
                    std_results[window] = std_val
                    stats_store[window] = {"mean": mean_val, "ci": ci_val}

            optimal_window = (
                min(std_results, key=std_results.get) if std_results else None
            )
            results[col] = {
                "optimal_window_size": optimal_window,
                "min_std": std_results.get(optimal_window),
                "mean": stats_store.get(optimal_window, {}).get("mean"),
                "ci": stats_store.get(optimal_window, {}).get("ci"),
            }
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
            }
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
            {column: {...fit results, n_target, additional_samples...}}
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
            }
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
            {column: ESS}
        """
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
        return results

