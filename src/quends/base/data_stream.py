import math

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.robust.scale import mad
from statsmodels.tsa.stattools import acf, adfuller


class DataStream:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def head(self, n=5):
        """Returns the first n rows of the DataFrame."""
        return self.df.head(n)

    def __len__(self):
        """Returns the length of this data stream."""
        return len(self.df)

    def variables(self):
        """Returns the variables in this data stream."""
        return self.df.columns

    # --- Helper methods to reduce duplicate code in statistical methods ---

    def _get_columns(self, column_name):
        """Return list of columns  based on input"""
        if column_name is None:
            return [col for col in self.df.columns if col != "time"]
        return [column_name] if isinstance(column_name, str) else column_name

    def _estimate_window(self, col, column_data, window_size):
        """Estimate the window size based on the effective sample size (ess)"""
        if window_size is None:
            ess_results = self.effective_sample_size(column_names=col)
            ess_value = ess_results.get(col, 10)  # Default ESS to 10 if not found
            return max(5, len(column_data) // ess_value)
        return window_size

    def _process_column(self, column_data, estimated_window, method):
        """
        Processed data (short-term averages) using the specified method.

        Args:
            column_data (pd.Series): Data for the column.
            estimated_window (int): Window size to use.
            method (str): Either "sliding" or "non-overlapping".

        Returns:
            pd.Series: Processed data.
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

    # ---------- Main Methods -------------

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
        Trims the data to start from the steady state and retains only the specified column.

        Args:
            column_name (str): Name of the column to analyze.
            window_size (int): Window size for analysis.
            start_time (float): Start time for the analysis.
            method (str): Method to use for steady state detection
                          ("std", "threshold", "rolling_variance").
            threshold (float): Threshold value for steady state detection (used with "threshold" and "rolling_variance" methods).
            robust (bool): Use median and MAD for non-normal data (used with "std" method).

        Returns:
            DataStream: A new DataStream instance containing the trimmed data.
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

        # print(steady_state_start_time)
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
        Find the steady-state start time based on standard deviation or robust median/MAD method.

        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
            column_name (str): Name of the column to analyze.
            window_size (int): Window size for analysis.
            start_time (float): Start time for the analysis.
            robust (bool): Use median and MAD for non-normal data.

        Returns:
            float: Steady state start time.
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
            # Compute central and scale values
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
        Find the steady-state start time based on rolling variance.

        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
            column_name (str): Name of the column to analyze.
            window_size (int): Window size for computing rolling variance.
            threshold (float): Multiplier for variance threshold determination.

        Returns:
            float: Steady state start time.
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
        """Normalize data excluding the 'time' column without modifying the original data stream."""
        scaler = MinMaxScaler()
        df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
        return df

    @staticmethod
    def find_steady_state_threshold(
        data, column_name, window_size, threshold, start_time=0.0
    ):
        """
        Find the steady-state start time based on a threshold method using a sliding window approach.

        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
            column_name (str): Name of the column to analyze.
            window_size (int): Window size for rolling averages.
            threshold (float): Threshold value for steady state detection.
            start_time (float): Start time for the analysis.

        Returns:
            float: Steady state start time.
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

        # rolling_avg = filtered[column_name].rolling(window=window_size).mean().dropna()
        rolling_std = filtered[column_name].rolling(window=window_size).std().rolling(
            window=window_size
        ).std() / np.sqrt(window_size)
        # Align indices between rolling_std and time
        common_idx = filtered.index.intersection(rolling_std.index)
        steady_state = filtered.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]

        if not steady_state.empty:
            return steady_state.iloc[0]
        return None

    def is_stationary(self, columns):
        """
        Check if the specified columns in the DataStream are stationary using the ADF test.

        Args:
            columns (str or list): Column name(s) to check for stationarity.

        Returns:
            dict: A dictionary with column names as keys and True/False as values.
                  True if stationary, False otherwise.
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
        Compute the effective sample size (ESS) for the data using the following steps:

        1. Determine the number of observations, n.
        2. Compute the autocorrelation function (ACF) for nlags = n/3.
        3. Calculate the two-tailed critical value and derive the confidence interval: CI = z_critical / sqrt(n).
        4. Identify significant lags (excluding lag 0) where the absolute ACF exceeds the CI.
        5. Sum the absolute ACF values at the significant lags.
        6. Compute ESS using: ESS = n / (1 + 2 * sum(|ACF| at significant lags)).

        Args:
            column_names (str or list, optional): Column(s) to compute ESS for.
                If None, all columns except 'time' are used.
            alpha (float): Significance level for the confidence interval (default 0.05).

        Returns:
            dict: A dictionary where keys are column names and values are the estimated ESS.
        """
        # Use all columns except 'time' if none specified.
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
            # Compute ACF with nlags = n/3 (converted to integer)
            nlags = int(n / 4)
            acf_values = acf(filtered, nlags=nlags)
            # Calculate the critical value for a two-tailed test.
            z_critical = norm.ppf(1 - alpha / 2)
            conf_interval = z_critical / np.sqrt(n)
            # Identify significant lags (excluding lag 0)
            significant_lags = np.where(np.abs(acf_values[1:]) > conf_interval)[0]
            # Sum the absolute autocorrelations at the significant lags
            acf_sum = np.sum(np.abs(acf_values[1:][significant_lags]))
            ESS = n / (1 + 2 * acf_sum)
            results[col] = int(np.ceil(ESS))
        return results

    def mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Compute the mean of the short-term averages.

        Args:
            column_name (str or list, optional): Column(s) to compute mean for. If None, infer columns.
            method (str): Method to calculate mean ("sliding" or "non-overlapping").
            window_size (int, optional): Window size for the selected method. If None, estimated from ESS.

        Returns:
            dict: Mean of the data for each column.
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
        Compute the uncertainty (standard error) of the mean of the short-term averages.

        Args:
            column_name (str or list, optional): Column(s) to compute standard deviation for. If None, infer columns.
            ddof (int): Delta degrees of freedom for standard deviation.
            method (str): Method to calculate uncertainty ("sliding" or "non-overlapping").
            window_size (int, optional): Window size for the selected method. If None, estimated from ESS.

        Returns:
            dict: Uncertainty of the mean for each column.
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

            # results[col] = {"mean uncertainty": np.std(proc_data, ddof=ddof)}
            results[col] = {"mean uncertainty": uncertainty}
        return results

    def confidence_interval(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Compute the confidence interval for the mean of the short-term averages using
        the mean and the uncertainty (standard error) computed from the short-term averages.

        Args:
            column_name (str or list, optional): Column(s) to compute confidence intervals for.
                If None, infer columns.
            ddof (int): Delta degrees of freedom for standard deviation.
            method (str): Method to calculate confidence intervals ("sliding" or "non-overlapping").
            window_size (int, optional): Window size for the selected method. If None, estimated from ESS.

        Returns:
            dict: Confidence interval for each column computed as:
                (mean - 1.96 * mean_uncertainty, mean + 1.96 * mean_uncertainty)
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
        Calculate mean, standard deviation, confidence interval, and plus-minus one standard deviation.

        Args:
            column_name (str or list, optional): Column(s) to compute statistics for. If None, infer columns.
            ddof (int): Delta degrees of freedom for standard deviation.
            method (str): Method to calculate statistics ("sliding" or "non-overlapping").
            window_size (int, optional): Window size for the selected method. If None, estimated from ESS.

        Returns:
            dict: Statistics for each column.
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
        Returns the optimal window size that results in the lowest uncertainty (minimum std) in the mean prediction.

        Args:
            column_name (str, optional): Column name to analyze. If None, infer the column.
            method (str): Method to calculate statistics ("sliding" or "non-overlapping").

        Returns:
            dict: Optimal window size, corresponding minimum std, mean, and confidence interval for each column.
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
        Compute cumulative statistics (cumulative mean, std, and standard error) for the data.

        Args:
            column_name (str or list, optional): Column(s) to compute cumulative statistics for. If None, infer columns.
            method (str): Method to calculate statistics ("sliding" or "non-overlapping").
            window_size (int, optional): Window size for the selected method. If None, estimated from ESS.

        Returns:
            dict: Cumulative statistics for each column.
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
        For each specified column, compute cumulative statistics on the processed data, fit a power‐law
        SEM model of the form:

            SEM(n) = A / (n^p)

        to the observed cumulative standard error (SEM) values, and then estimate how many additional
        samples are required to reduce the current SEM by the given reduction factor.

        Args:
            column_name (str or list, optional): Column(s) to compute additional samples for.
                If None, use all columns except 'time'.
            ddof (int): Degrees of freedom for standard deviation.
            method (str): Processing method ("sliding" or "non-overlapping").
            window_size (int, optional): Window size for processing data.
            reduction_factor (float): Fraction by which to reduce the current SEM
                (e.g. 0.1 for a 10% reduction; target SEM = (1 - reduction_factor) * current SEM).

        Returns:
            dict: For each column, a dictionary containing:
                - 'A_est': estimated A parameter,
                - 'p_est': estimated exponent p,
                - 'n_current': current number of samples (from cumulative statistics),
                - 'current_sem': current SEM (at n_current),
                - 'target_sem': target SEM,
                - 'n_target': estimated total samples required,
                - 'additional_samples': additional samples needed (n_target - n_current).
        """
        # Get cumulative statistics computed on the processed data (using your cumulative_statistics method)
        stats = self.cumulative_statistics(
            column_name, method=method, window_size=window_size
        )
        results = {}

        # Get the list of columns to process
        columns = self._get_columns(column_name)

        for col in columns:
            if "cumulative_uncertainty" not in stats.get(col, {}):
                results[col] = {"error": f"No cumulative SEM data for column '{col}'"}
                continue

            # Estimate window size used for processing
            column_data = self.df[col].dropna()
            est_win = self._estimate_window(col, column_data, window_size)

            # Convert cumulative uncertainty to a NumPy array
            cum_sem = np.array(stats[col]["cumulative_uncertainty"])
            n_current = len(cum_sem)
            cumulative_count = np.arange(1, n_current + 1)

            # Remove any NaN or infinite values (typically the first sample might be NaN)
            mask = np.isfinite(cum_sem)
            valid_count = cumulative_count[mask]
            valid_sem = cum_sem[mask]

            if len(valid_count) < 2:
                results[col] = {"error": "Not enough valid data points for fitting."}
                continue

            # Estimate window size used for processing
            column_data = self.df[col].dropna()
            est_win = self._estimate_window(col, column_data, window_size)

            # Define the power-law SEM model: SEM(n) = A / (n^p)
            def power_law_model(n, A, p):
                return A / (n**p)

            # Fit the model to the observed SEM data with an initial guess.
            popt, _ = curve_fit(power_law_model, valid_count, valid_sem, p0=[1.0, 0.5])
            A_est, p_est = popt
            p_est = abs(p_est)

            # Use the full data length as the current sample count.
            # current_sem = valid_sem[-1]
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
        .. |ACF| replace:: Autocorrelation Function

        Compute the effective sample size (ESS) for each specified column using a weighted
        autocorrelation approach where "significant" lags are defined as all lags up to (but not including)
        the first lag at which the absolute ACF drops below the 95% confidence interval.

        The steps are:

        1. Determine the number of observations, n.
        2. Compute the autocorrelation function (ACF) for nlags = int(n/3).
        3. Calculate the two-tailed critical value:
            z_critical = norm.ppf(1 - alpha/2),
            and then the confidence interval:
            conf_interval = z_critical / sqrt(n).
        4. Find the first lag (excluding lag 0) where |ACF| < conf_interval.
        5. Sum the absolute ACF values for all lags before that drop.
        6. Compute ESS using:
            ESS = n / (1 + 2 * sum(|ACF| at lags before drop))

        Args:
            column_names (str or list, optional): Column(s) for which to compute ESS.
                If None, all columns except 'time' are used.
            alpha (float): Significance level for the confidence interval (default 0.05).

        Returns:
            dict: A dictionary with keys as column names and values as the estimated ESS.
        """
        # Use all columns except 'time' if none specified.
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
            # Use nlags = int(n/3) for ACF computation.
            nlags = int(n / 3)
            acf_values = acf(filtered, nlags=nlags, fft=False)
            # Compute critical value for a two-tailed test.
            z_critical = norm.ppf(1 - alpha / 2)
            conf_interval = z_critical / np.sqrt(n)

            # Find the first lag (excluding lag 0) where |ACF| < conf_interval.
            significant_idx = None
            for i in range(1, len(acf_values)):
                if np.abs(acf_values[i]) < conf_interval:
                    significant_idx = i
                    break
            if significant_idx is None:
                # If no such lag is found, consider all lags except lag 0.
                significant_lags = np.arange(1, len(acf_values))
            else:
                significant_lags = np.arange(1, significant_idx)

            # Sum the absolute ACF values for the significant lags.
            acf_sum = np.sum(np.abs(acf_values[significant_lags]))
            ESS = n / (1 + 2 * acf_sum)
            results[col] = int(np.floor(ESS))
        return results
