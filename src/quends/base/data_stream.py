import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm, rankdata
from sklearn.preprocessing import MinMaxScaler
from statsmodels.robust.scale import mad
from statsmodels.tsa.stattools import acf, adfuller

<<<<<<< HEAD
<<<<<<< HEAD

def deduplicate_history(history):
    """
    Remove duplicate operations from a history list, keeping only the most recent occurrence of each operation.

    Scans the history of operations (each represented as a dict with at least an 'operation' key)
    from end to start, retaining only the last entry for each unique operation name while preserving
    the overall order of those final occurrences.

    Parameters
    ----------
    history : list of dict
        Each dict must contain:
          - 'operation': str, the name of the operation.
          - additional keys for operation-specific metadata (e.g., 'options').

    Returns
    -------
    list of dict
        A filtered list with only the final entry of each operation, ordered as in the original list.
    """
    seen = set()
    out = []
    # Reverse to keep last call
    for entry in reversed(history):
        op = entry["operation"]
        if op not in seen:
            out.append(entry)
            seen.add(op)
    return list(reversed(out))

def to_native_types(obj):
    """
    Recursively convert NumPy scalar and array types in nested structures to native Python types.

    This function walks through dictionaries, lists, tuples, NumPy scalars, and arrays,
    converting them into Python built-ins:

    - NumPy scalar → Python int or float
    - NumPy array  → Python list (recursively)

    Parameters
    ----------
    obj : any
        The object to convert. Supported container types are dict, list, tuple,
        NumPy ndarray/scalar. Other types are returned unchanged.

    Returns
    -------
    any
        A new object mirroring the input structure but with all NumPy data types replaced
        by their native Python equivalents.
    """
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t([to_native_types(v) for v in obj])
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj


class DataStream:
    """
<<<<<<< HEAD
    A pipeline for time-series and simulation trace analysis with provenance tracking.

    DataStream encapsulates a pandas DataFrame with a required 'time' column and any number of
    signal columns.  All analysis methods record their operation name and options in an internal
    history, and returned results include deduplicated metadata lineage.

    Core features include:
    - Stationarity testing and steady-state trimming via multiple methods.
    - Statistical summaries: means, uncertainties, confidence intervals, and effective sample size (ESS).
    - Robust ESS estimation using rank-based and pairwise correlation techniques.
    - Incremental and cumulative statistics, plus sample-size planning via power-law fits.

    Attributes
    ----------
    df : pandas.DataFrame
        The underlying time-series data, with 'time' as one column.
    _history : list of dict
        Records of all operations performed, including their options.
    """

    def __init__(self, df: pd.DataFrame, _history=None):
        """
        Initialize a DataStream.
=======
    A wrapper for time-series data (pandas DataFrame) that provides rich statistical and
    steady-state analysis for simulation and experimental outputs.
=======
class DataStream:
    """
    Scientific time-series and simulation trace analysis with full pipeline reproducibility.
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)
=======
def deduplicate_history(history):
    """
    Remove duplicate operations from a history list, keeping only the most recent occurrence of each operation.
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)

    Scans the history of operations (each represented as a dict with at least an 'operation' key)
    from end to start, retaining only the last entry for each unique operation name while preserving
    the overall order of those final occurrences.

    Parameters
    ----------
    history : list of dict
        Each dict must contain:
          - 'operation': str, the name of the operation.
          - additional keys for operation-specific metadata (e.g., 'options').

    Returns
    -------
    list of dict
        A filtered list with only the final entry of each operation, ordered as in the original list.
    """
    seen = set()
    out = []
    # Reverse to keep last call
    for entry in reversed(history):
        op = entry['operation']
        if op not in seen:
            out.append(entry)
            seen.add(op)
    return list(reversed(out))

def to_native_types(obj):
    """
    Recursively convert NumPy scalar and array types in nested structures to native Python types.

    This function walks through dictionaries, lists, tuples, NumPy scalars, and arrays,
    converting them into Python built-ins:
      - NumPy scalar → Python int or float
      - NumPy array  → Python list (recursively)

    Parameters
    ----------
    obj : any
        The object to convert. Supported container types are dict, list, tuple,
        NumPy ndarray/scalar. Other types are returned unchanged.

    Returns
    -------
    any
        A new object mirroring the input structure but with all NumPy data types replaced
        by their native Python equivalents.
    """
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t([to_native_types(v) for v in obj])
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj

class DataStream:
    """
    A pipeline for time-series and simulation trace analysis with provenance tracking.

    DataStream encapsulates a pandas DataFrame with a required 'time' column and any number of
    signal columns.  All analysis methods record their operation name and options in an internal
    history, and returned results include deduplicated metadata lineage.

    Core features include:
    - Stationarity testing and steady-state trimming via multiple methods.
    - Statistical summaries: means, uncertainties, confidence intervals, and effective sample size (ESS).
    - Robust ESS estimation using rank-based and pairwise correlation techniques.
    - Incremental and cumulative statistics, plus sample-size planning via power-law fits.

    Attributes
    ----------
    df : pandas.DataFrame
        The underlying time-series data, with 'time' as one column.
    _history : list of dict
        Records of all operations performed, including their options.
    """

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, df: pd.DataFrame):
        """
        Initialize a DataStream object.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
=======
    def __init__(self, df: pd.DataFrame, _history=None):):
=======
    def __init__(self, df: pd.DataFrame, _history=None):
>>>>>>> 26c7740 (Fix syntax error: remove unmatched parenthesis in __init__)
        """
        Initialize a DataStream.
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)

        Parameters
        ----------
        df : pandas.DataFrame
<<<<<<< HEAD
<<<<<<< HEAD
            Must contain a 'time' column and one or more signal columns.
        _history : list of dict, optional
            Existing operation history to inherit.  If None, starts empty.
=======
            DataFrame containing 'time' and data columns.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        self.df = df
        self._history = list(_history) if _history is not None else []
=======
    def __init__(self, df: pd.DataFrame, _history=None):
=======
            Must contain a 'time' column and one or more signal columns.
        _history : list of dict, optional
            Existing operation history to inherit.  If None, starts empty.
        """
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)
        self.df = df
        self._history = list(_history) if _history is not None else []

    def _add_history(self, operation, options):
        """
        Record an operation and its options into the internal history.

        Private helper; not intended for external use.
        """
        options = {k: v for k, v in options.items() if k not in ('self', 'cls', '__class__')}
        self._history.append({
            "operation": operation,
            "options": options
        })
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)

    def _add_history(self, operation, options):
        """
        Record an operation and its options into the internal history.

        Private helper; not intended for external use.
        """
        options = {k: v for k, v in options.items() if k not in ('self', 'cls', '__class__')}
        self._history.append({
            "operation": operation,
            "options": options
        })
        
    def get_metadata(self):
        """
        Return the deduplicated operation history for this DataStream.
        Returns
        -------
            list of dict
            The deduplicated operation history, with options for each operation.
        """
        return deduplicate_history(self._history)
    
    
    def head(self, n=5):
<<<<<<< HEAD
<<<<<<< HEAD
        """
<<<<<<< HEAD
        Return the first `n` rows of the underlying DataFrame.
=======
        Return the first n rows of the underlying DataFrame.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

        Parameters
        ----------
        n : int, optional
<<<<<<< HEAD
            Number of rows to return. Defaults to 5.
=======
            Number of rows to return (default: 5).
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
=======
        """
        Return the first `n` rows of the underlying DataFrame.

        Parameters
        ----------
        n : int, default=5
            Number of rows to return.
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)

        Returns
        -------
        pandas.DataFrame
<<<<<<< HEAD
<<<<<<< HEAD
            The first `n` rows of the DataFrame.
=======
            The first n rows.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        return self.df.head(n)

    def __len__(self):
        """
<<<<<<< HEAD
        Return the number of rows in the DataStream.
=======
        Return the length (number of rows) of the DataStream.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

        Returns
        -------
        int
<<<<<<< HEAD
            Row count of `self.df`.
=======
            Number of data points (rows).
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        return len(self.df)

    def variables(self):
        """
<<<<<<< HEAD
        List the signal variable (column) names, excluding the 'time' column.

        Returns
        -------
        Index
            ColumnIndex of variable names in `self.df`.
        """
        return self.df.columns

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
        Trim the DataStream to its steady-state portion based on a chosen detection method.
        Always returns a DataStream (possibly empty if trim fails), with operation metadata
        and any messages stored in the _history attribute.
=======
        List the variable (column) names in the DataStream.

        Returns
        -------
        pandas.Index
            Names of the columns in the DataFrame.
        """
        return self.df.columns

    # --- Internal Helper Methods ---
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

        Parameters
        ----------
        column_name : str
            Name of the signal column to analyze for steady-state.
        batch_size : int, default=10
            Window size for steady-state detection.
        start_time : float, default=0.0
            Earliest time to consider in the analysis.
        method : {'std', 'threshold', 'rolling_variance'}, default='std'
            Detection method:
            - 'std': sliding std-based criteria (requires stationarity).
            - 'threshold': rolling-std threshold (requires `threshold`).
            - 'rolling_variance': comparison to mean variance times `threshold`.
        threshold : float or None
            Threshold value for the 'threshold' or 'rolling_variance' methods.
        robust : bool, default=True
            Use median/MAD instead of mean/std for the 'std' method.

        Returns
        -------
        DataStream
            New DataStream containing the trimmed data, or empty if trimming failed.
            Operation metadata and any messages are in the ._history attribute.
        """
        # Check for stationarity
        stationary_result = self.is_stationary(column_name)
        is_stat = (
            stationary_result.get(column_name, False)
            if isinstance(stationary_result, dict)
            else bool(stationary_result)
        )
        new_history = self._history.copy()
        options = dict(
            column_name=column_name,
            batch_size=batch_size,
            start_time=start_time,
            method=method,
            threshold=threshold,
            robust=robust,
        )
        if not is_stat:
            options["message"] = (
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data."
            )
            new_history.append({"operation": "trim", "options": options})
            # Return a DataStream with an empty dataframe, but history is preserved
            empty_df = self.df.iloc[0:0].copy()
            return DataStream(empty_df, _history=new_history)

        # Preprocess
        data = self.df[self.df["time"] >= start_time].reset_index(drop=True)
        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
            data = data.loc[non_zero_index:].reset_index(drop=True)

        # Steady-state detection
        if method == "std":
            steady_state_start_time = self.find_steady_state_std(
                data, column_name, window_size=batch_size, robust=robust
            )
        elif method == "threshold":
            if threshold is None:
                options["message"] = (
                    "Threshold must be specified for the 'threshold' method."
                )
                new_history.append({"operation": "trim", "options": options})
                return DataStream(self.df.iloc[0:0].copy(), _history=new_history)
            steady_state_start_time = self.find_steady_state_threshold(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        elif method == "rolling_variance":
            threshold = threshold if threshold is not None else 0.1
            steady_state_start_time = self.find_steady_state_rolling_variance(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        else:
            options["message"] = (
                "Invalid method. Choose 'std', 'threshold', or 'rolling_variance'."
            )
            new_history.append({"operation": "trim", "options": options})
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

        options["sss_start"] = steady_state_start_time
        if steady_state_start_time is not None:
            trimmed_df = self.df.loc[
                self.df["time"] >= steady_state_start_time, ["time", column_name]
            ].reset_index(drop=True)
            new_history.append({"operation": "trim", "options": options})
            return DataStream(trimmed_df, _history=new_history)
        else:
            options["message"] = (
                f"Steady-state start time could not be determined for column '{column_name}'."
            )
            new_history.append({"operation": "trim", "options": options})
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

    @staticmethod
    def find_steady_state_std(data, column_name, window_size=10, robust=True):
        """
        Identify the earliest time point when the signal remains within ±1/2/3σ proportions.

        Parameters
        ----------
        data : DataFrame
            Subset of the original df (must include 'time' and signal column).
        column_name : str
        window_size : int
            Number of samples to evaluate the steady-state criteria.
        robust : bool
            If True, use median and MAD; else mean and std.

        Returns
        -------
        float or None
            Detected start time of steady-state, or None if not found.
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
        Detect steady-state when rolling variance falls below a fraction of its mean.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Fraction of mean rolling std below which to consider steady-state.

        Returns
        -------
        float or None
            Time of first below-threshold variance, or None.
        """
        ts = data[["time", column_name]].dropna()
        time_values = ts["time"]
        signal_values = ts[column_name]
        rolling_variance = signal_values.rolling(window=window_size).std()
        threshold_val = rolling_variance.mean() * threshold
        steady_state_index = np.where(rolling_variance < threshold_val)[0]
        if len(steady_state_index) > 0:
            return time_values.iloc[steady_state_index[0]]
        return None

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

    @staticmethod
    def find_steady_state_threshold(data, column_name, window_size, threshold):
        """
        Use rolling standard deviation on normalized data to detect steady-state.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Std threshold under which to mark steady-state.

        Returns
        -------
        float or None
        """
        normalized_data = DataStream.normalize_data(data.copy())
        time_series = normalized_data[["time", column_name]]
        if len(time_series) < window_size:
            return None
        rolling_std = time_series[column_name].rolling(window=window_size).std()
        common_idx = time_series.index.intersection(rolling_std.index)
        steady_state = time_series.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]
        if not steady_state.empty:
            return steady_state.iloc[0]
        return None

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
        self._add_history(
            "effective_sample_size", {"column_names": column_names, "alpha": alpha}
        )
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
        metadata = deduplicate_history(self._history)
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
        self._add_history(
            "ess_robust",
            {
                "column_names": column_names,
                "rank_normalize": rank_normalize,
                "min_samples": min_samples,
                "return_relative": return_relative,
            },
        )
        if column_names is None:
            column_names = [col for col in self.df.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
        results = {}
        for col in column_names:
            if col not in self.df.columns:
                results[col] = {"error": f"Column '{col}' not found."}
                continue
            x = self.df[col].dropna().values
            ess = self.robust_effective_sample_size(
                x,
                rank_normalize=rank_normalize,
                min_samples=min_samples,
                return_relative=return_relative,
            )
            results[col] = ess
        metadata = deduplicate_history(self._history)
        return {"results": to_native_types(results), "metadata": metadata}

    # --------- Statistical summaries (unchanged except metadata) --------
    def _mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Compute block or sliding window means for each column.

        Private helper for compute_statistics and confidence intervals.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
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

    def _mean_uncertainty(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Estimate the standard error of the mean via block/sliding windows.

        Private helper.
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
            uncertainty = float(np.std(proc_data, ddof=ddof) / np.sqrt(effective_n))
            results[col] = {
                "mean_uncertainty": uncertainty,
                "window_size": int(est_win),
            }
        return results

    def _confidence_interval(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
        """
        Build 95% confidence intervals around block/sliding means.

        Private helper.
        """
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
        mean_results = self._mean(column_name, method=method, window_size=window_size)
        mu_results = self._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ci_results = self._confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ess_dict = self.effective_sample_size(column_names=column_name)
        for col in columns:
            column_data = self.df[col].dropna()
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
        op_options = dict(
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
        )
        full_history = self._history.copy()
        full_history.append({"operation": "compute_statistics", "options": op_options})
        statistics["metadata"] = deduplicate_history(full_history)
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
                "window_size": int(est_win),
            }
        op_options = dict(
            column_name=column_name, method=method, window_size=window_size
        )
        full_history = self._history.copy()
        full_history.append(
            {"operation": "cumulative_statistics", "options": op_options}
        )
        results["metadata"] = deduplicate_history(full_history)
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
                "A_est": float(A_est),
                "p_est": float(p_est),
                "n_current": int(n_current),
                "current_sem": float(current_sem),
                "target_sem": float(target_sem),
                "n_target": float(n_target),
                "additional_samples": int(math.ceil(additional_samples)),
                "window_size": int(est_win),
            }
        op_options = dict(
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
            reduction_factor=reduction_factor,
        )
        full_history = self._history.copy()
        full_history.append({"operation": "additional_data", "options": op_options})
        results["metadata"] = deduplicate_history(full_history)
        return to_native_types(results)

    # ------ Helper functions --------
    def _get_columns(self, column_name):
        """
<<<<<<< HEAD
        Resolve `column_name` parameter into a list of valid DataFrame columns.
=======
        Infer a list of columns to operate on based on user input.

        Parameters
        ----------
        column_name : str, list, or None
            Single column name, list of names, or None (use all except 'time').
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

        Returns
        -------
        list of str
<<<<<<< HEAD
=======
            The column names to process.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        if column_name is None:
            return [col for col in self.df.columns if col != "time"]
        return [column_name] if isinstance(column_name, str) else column_name

    def _estimate_window(self, col, column_data, window_size):
        """
<<<<<<< HEAD
        Determine block size: either user-provided or derived from ESS.

        Ensures a minimum window of 5 samples.
=======
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
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        if window_size is None:
            # Get ESS dictionary from method
            ess_results = self.effective_sample_size(column_names=col)
<<<<<<< HEAD
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
=======
            ess_value = ess_results.get(col, 10)
            return max(5, len(column_data) // ess_value)
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        return window_size

    # def _estimate_window(self, col, column_data, window_size):
    #    if window_size is None:
    #        ess_results = self.effective_sample_size(column_names=col)
    #        ess_value = ess_results["results"].get(col, 10)
    #        return max(5, len(column_data) // ess_value)
    #    return window_size

    def _process_column(self, column_data, estimated_window, method):
        """
<<<<<<< HEAD
        Transform a 1D series into block or sliding window means.
=======
        Apply windowed averaging (sliding or non-overlapping) to a column.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

        Parameters
        ----------
        column_data : pandas.Series
<<<<<<< HEAD
        estimated_window : int
        method : {'sliding', 'non-overlapping'}
=======
            Data to process.
        estimated_window : int
            Window size to use.
        method : {"sliding", "non-overlapping"}
            Windowing approach.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

        Returns
        -------
        pandas.Series
<<<<<<< HEAD
=======
            Series of window means.
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
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

<<<<<<< HEAD
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
=======
    # =======================
    #    Main Functionality
    # =======================
=======
        """Return the first n rows of the underlying DataFrame."""
=======
            The first `n` rows of `self.df`.
        """
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)
        return self.df.head(n)

    def __len__(self):
        """
        Return the number of rows in the DataStream.

        Returns
        -------
        int
            Row count of `self.df`.
        """
        return len(self.df)

    def variables(self):
        """
        List the signal variable (column) names, excluding the 'time' column.

<<<<<<< HEAD
    # ---------- Main Trim and Detection Logic ----------
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)
=======
        Returns
        -------
        Index
            ColumnIndex of variable names in `self.df`.
        """
        return self.df.columns
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)

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
<<<<<<< HEAD
        Trim the DataStream to start from steady-state using a chosen method.
<<<<<<< HEAD
=======
        Handles all start time logic (user-specified and auto-skip to first nonzero).
        Tracks options for reproducibility.
>>>>>>> 44a901e (Improvement of the updated the trimin  data_stream.py)
=======
        Trim the DataStream to its steady-state portion based on a chosen detection method.

        Records the trim operation in history and returns a dict containing:
          - 'results': a new DataStream of trimmed data or None if trimming failed.
          - 'metadata': deduplicated operation lineage.
          - optionally 'message' on failure.
>>>>>>> e66b052 (Update: improvements to data_stream, ensemble, and exporter modules)

        Parameters
        ----------
        column_name : str
            Name of the signal column to analyze for steady-state.
        batch_size : int, default=10
            Window size for steady-state detection.
        start_time : float, default=0.0
            Earliest time to consider in the analysis.
        method : {'std', 'threshold', 'rolling_variance'}, default='std'
            Detection method:
              - 'std': sliding std-based criteria (requires stationarity).
              - 'threshold': rolling-std threshold (requires `threshold`).
              - 'rolling_variance': comparison to mean variance times `threshold`.
        threshold : float or None
            Threshold value for the 'threshold' or 'rolling_variance' methods.
        robust : bool, default=True
            Use median/MAD instead of mean/std for the 'std' method.

        Returns
        -------
<<<<<<< HEAD
        DataStream
            New DataStream with trimmed data, from steady-state onward.
<<<<<<< HEAD
        """
=======

        Raises
        ------
        ValueError
            If the data column is not stationary, or if no steady-state found.
=======
        dict
            {
              'results': DataStream or None,
              'metadata': list of dict,
              'message': str (if occurred)
            }
>>>>>>> e66b052 (Update: improvements to data_stream, ensemble, and exporter modules)
        """
        stationary_result = self.is_stationary(column_name)
        is_stat = stationary_result.get(column_name, False) if isinstance(stationary_result, dict) else bool(stationary_result)
        new_history = self._history.copy()
        options = dict(
            column_name=column_name, batch_size=batch_size, start_time=start_time,
            method=method, threshold=threshold, robust=robust,
        )
        if not is_stat:
            options["message"] = (
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data."
            )
            new_history.append({"operation": "trim", "options": options})
            return {"results": None, "metadata": deduplicate_history(new_history), "message": options["message"]}
            #return {"results": None, "metadata": deduplicate_history(new_history)}

        data = self.df[self.df["time"] >= start_time].reset_index(drop=True)
        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
            data = data.loc[non_zero_index:].reset_index(drop=True)

>>>>>>> 44a901e (Improvement of the updated the trimin  data_stream.py)
        if method == "std":
            steady_state_start_time = self.find_steady_state_std(
                data, column_name, window_size=batch_size, robust=robust
            )
        elif method == "threshold":
            if threshold is None:
                options["message"] = "Threshold must be specified for the 'threshold' method."
                new_history.append({"operation": "trim", "options": options})
                return {"results": None, "metadata": deduplicate_history(new_history)}
            steady_state_start_time = self.find_steady_state_threshold(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        elif method == "rolling_variance":
            threshold = threshold if threshold is not None else 0.1
            steady_state_start_time = self.find_steady_state_rolling_variance(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        else:
            options["message"] = "Invalid method. Choose 'std', 'threshold', or 'rolling_variance'."
            new_history.append({"operation": "trim", "options": options})
            return {"results": None, "metadata": deduplicate_history(new_history)}

        options['sss_start'] = steady_state_start_time
        if steady_state_start_time is not None:
            trimmed_df = self.df.loc[
                self.df["time"] >= steady_state_start_time, ["time", column_name]
            ].reset_index(drop=True)
            new_history.append({"operation": "trim", "options": options})
            return {
                "results": DataStream(trimmed_df, _history=new_history),
                "metadata": deduplicate_history(new_history)
            }
        else:
            options["message"] = (
                f"Steady-state start time could not be determined for column '{column_name}'."
            )
            new_history.append({"operation": "trim", "options": options})
            return {"results": None, "metadata": deduplicate_history(new_history), "message": options["message"]}
            #return {"results": None, "metadata": deduplicate_history(new_history)}

    @staticmethod
    def find_steady_state_std(
        data, column_name, window_size=10, robust=True
    ):
        """
        Identify the earliest time point when the signal remains within ±1/2/3σ proportions.

        Parameters
        ----------
        data : DataFrame
            Subset of the original df (must include 'time' and signal column).
        column_name : str
        window_size : int
            Number of samples to evaluate the steady-state criteria.
        robust : bool
            If True, use median and MAD; else mean and std.

        Returns
        -------
        float or None
            Detected start time of steady-state, or None if not found.
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
        Detect steady-state when rolling variance falls below a fraction of its mean.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Fraction of mean rolling std below which to consider steady-state.

        Returns
        -------
        float or None
            Time of first below-threshold variance, or None.
        """
        ts = data[["time", column_name]].dropna()
        time_values = ts["time"]
        signal_values = ts[column_name]
        rolling_variance = signal_values.rolling(window=window_size).std()
        threshold_val = rolling_variance.mean() * threshold
        steady_state_index = np.where(rolling_variance < threshold_val)[0]
        if len(steady_state_index) > 0:
            return time_values.iloc[steady_state_index[0]]
        return None

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

    @staticmethod
    def find_steady_state_threshold(
        data, column_name, window_size, threshold
    ):
        """
        Use rolling standard deviation on normalized data to detect steady-state.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Std threshold under which to mark steady-state.

        Returns
        -------
        float or None
        """
        normalized_data = DataStream.normalize_data(data.copy())
        time_series = normalized_data[["time", column_name]]
        if len(time_series) < window_size:
            return None
        rolling_std = time_series[column_name].rolling(window=window_size).std()
        common_idx = time_series.index.intersection(rolling_std.index)
        steady_state = time_series.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]
        if not steady_state.empty:
            return steady_state.iloc[0]
        return None

<<<<<<< HEAD
    # ============= Statistical Methods with Option Tracking ==============

<<<<<<< HEAD
        Parameters
        ----------
        columns : str or list of str
            Name(s) of columns to test.

        Returns
        -------
        dict
            {column: True/False (stationary), or error message}
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        self._add_history("is_stationary", {"columns": columns})
        if isinstance(columns, str):
            columns = [columns]
        results = {}
        for column in columns:
            try:
                p_value = adfuller(self.df[column].dropna(), autolag="AIC")[1]
                results[column] = p_value < 0.05
            except Exception as e:
                results[column] = f"Error: {e}"
        return results

<<<<<<< HEAD
    # === Compatibility wrappers for legacy tests ===

    def mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Legacy wrapper for test compatibility. Returns only mean (not dict).
=======
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

    # ========== Private Statistical Helpers ==========
    def _mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
<<<<<<< HEAD
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
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        results = self._mean(
            column_name=column_name, method=method, window_size=window_size
        )
        if column_name is None:
            # Return dict of means for all columns
            return {col: val["mean"] for col, val in results.items() if "mean" in val}
        col = column_name if isinstance(column_name, str) else list(results.keys())[0]
        return results[col]["mean"]
=======
        Helper for windowed mean calculation (private).
        """
=======
    def _mean(self, column_name=None, method="non-overlapping", window_size=None):
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)
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
>>>>>>> 64326bf (Improvement of the docstring in  data_stream.py)

    def _mean_uncertainty(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
<<<<<<< HEAD
        """
<<<<<<< HEAD
<<<<<<< HEAD
        Legacy wrapper for test compatibility. Returns only mean_uncertainty (not dict).
        """
        results = self._mean_uncertainty(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size
        )
        if column_name is None:
            return {
                col: val["mean_uncertainty"]
                for col, val in results.items()
                if "mean_uncertainty" in val
            }
        col = column_name if isinstance(column_name, str) else list(results.keys())[0]
        return results[col]["mean_uncertainty"]
=======
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
=======
        Helper for standard error of windowed mean (private).
>>>>>>> 64326bf (Improvement of the docstring in  data_stream.py)
        """
=======
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)
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
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)

    def _confidence_interval(
        self, column_name=None, ddof=1, method="non-overlapping", window_size=None
    ):
<<<<<<< HEAD
        """
<<<<<<< HEAD
<<<<<<< HEAD
        Legacy wrapper for test compatibility. Returns only CI tuple.
=======
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
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
        """
        results = self._confidence_interval(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size
=======
        Helper for 95% confidence interval of windowed mean (private).
        """
=======
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)
        results = {}
        mean_results = self._mean(column_name, method=method, window_size=window_size)
        uncertainty_results = self._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
>>>>>>> 64326bf (Improvement of the docstring in  data_stream.py)
        )
<<<<<<< HEAD
        if column_name is None:
            return {
                col: val["confidence_interval"]
                for col, val in results.items()
                if "confidence_interval" in val
=======

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
<<<<<<< HEAD
                "pm_std": (
                    mean_results[col]["mean"] - mu_results[col]["mean uncertainty"],
                    mean_results[col]["mean"] + mu_results[col]["mean uncertainty"],
                ),
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
=======
                "pm_std": (mean_val - mean_uncertainty, mean_val + mean_uncertainty),
                "effective_sample_size": ess_val,
                "window_size": est_win,
>>>>>>> 64326bf (Improvement of the docstring in  data_stream.py)
            }
<<<<<<< HEAD
        col = column_name if isinstance(column_name, str) else list(results.keys())[0]
        return results[col]["confidence_interval"]

    def optimal_window_size(self, method="sliding"):
        """
<<<<<<< HEAD
        Stub for compatibility. Return a default or best-guess window size.
        """
        # Just return a default for now (since the real logic is probably more complex)
        return 1

    def effective_sample_size_below(self, column_names=None, alpha=0.05):
        """
        Stub for compatibility with legacy test. Returns dummy value.
        """
        # We could implement a real one if needed; for now, return 0 for all columns.
=======
        Find the window size that minimizes mean uncertainty (std) for a column.
=======

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
>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)

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

=======
    # ----------- ESS (classic and robust) ----------------
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)
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
        metadata = deduplicate_history(self._history)
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

    def ess_robust(self, column_names=None, rank_normalize=False, min_samples=8, return_relative=False):
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
        self._add_history("ess_robust", {
            "column_names": column_names,
            "rank_normalize": rank_normalize,
            "min_samples": min_samples,
            "return_relative": return_relative,
        })
        if column_names is None:
            column_names = [col for col in self.df.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
        results = {}
        for col in column_names:
            if col not in self.df.columns:
                results[col] = {"error": f"Column '{col}' not found."}
                continue
            x = self.df[col].dropna().values
            ess = self.robust_effective_sample_size(x, rank_normalize=rank_normalize, min_samples=min_samples, return_relative=return_relative)
            results[col] = ess
        metadata = deduplicate_history(self._history)
        return {"results": to_native_types(results), "metadata": metadata}

    # --------- Statistical summaries (unchanged except metadata) --------
    def _mean(self, column_name=None, method="non-overlapping", window_size=None):

        """
        Compute block or sliding window means for each column.

        Private helper for compute_statistics and confidence intervals.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            proc_data = self._process_column(column_data, est_win, method)
            results[col] = {"mean": float(np.mean(proc_data)), "window_size": int(est_win)}
        return results

    def _mean_uncertainty(self, column_name=None, ddof=1, method="non-overlapping", window_size=None):
        """
        Estimate the standard error of the mean via block/sliding windows.

        Private helper.
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
            uncertainty = float(np.std(proc_data, ddof=ddof) / np.sqrt(effective_n))
            results[col] = {"mean_uncertainty": uncertainty, "window_size": int(est_win)}
        return results

    def _confidence_interval(self, column_name=None, ddof=1, method="non-overlapping", window_size=None):
        """
        Build 95% confidence intervals around block/sliding means.

        Private helper.
        """
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
        mean_results = self._mean(column_name, method=method, window_size=window_size)
        mu_results = self._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ci_results = self._confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ess_dict = self.effective_sample_size(column_names=column_name)
        for col in columns:
            column_data = self.df[col].dropna()
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
        op_options = dict(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size,
        )
        full_history = self._history.copy()
        full_history.append({"operation": "compute_statistics", "options": op_options})
        statistics["metadata"] = deduplicate_history(full_history)
        return to_native_types(statistics)

    def cumulative_statistics(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Generate cumulative mean and uncertainty time series for each column.

        Records operation and returns per-column cumulative arrays plus window_size.
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
                "window_size": int(est_win),
            }
        op_options = dict(
            column_name=column_name, method=method, window_size=window_size
        )
        full_history = self._history.copy()
        full_history.append({"operation": "cumulative_statistics", "options": op_options})
        results["metadata"] = deduplicate_history(full_history)
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
                "A_est": float(A_est),
                "p_est": float(p_est),
                "n_current": int(n_current),
                "current_sem": float(current_sem),
                "target_sem": float(target_sem),
                "n_target": float(n_target),
                "additional_samples": int(math.ceil(additional_samples)),
                "window_size": int(est_win),
            }
        op_options = dict(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size, reduction_factor=reduction_factor
        )
        full_history = self._history.copy()
        full_history.append({"operation": "additional_data", "options": op_options})
        results["metadata"] = deduplicate_history(full_history)
        return to_native_types(results)

    # ------ Helper functions --------
    def _get_columns(self, column_name):
        """
        Resolve `column_name` parameter into a list of valid DataFrame columns.

        Returns
        -------
        list of str
        """
        if column_name is None:
            return [col for col in self.df.columns if col != "time"]
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
                ess_val = ess_results.get(col, 10) if isinstance(ess_results, dict) else 10
            # Avoid division by zero or negative
            try:
                ess_val = max(1, int(round(ess_val)))
            except Exception:
                ess_val = 10
            return max(5, len(column_data) // ess_val)
        return window_size



    #def _estimate_window(self, col, column_data, window_size):
    #    if window_size is None:
    #        ess_results = self.effective_sample_size(column_names=col)
    #        ess_value = ess_results["results"].get(col, 10)
    #        return max(5, len(column_data) // ess_value)
    #    return window_size

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
            step_size = max(1, estimated_window) #max(1, estimated_window // 4)
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
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
=======
        self._add_history("effective_sample_size_below", {"column_names": column_names, "alpha": alpha})

>>>>>>> accec82 (Improvement of the updated the trimin  data_stream.py)
        if column_names is None:
            column_names = [col for col in self.df.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
<<<<<<< HEAD
        return {col: 0 for col in column_names}
=======

=======
        self._add_history("is_stationary", {"columns": columns})
        if isinstance(columns, str):
            columns = [columns]
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)
        results = {}
        for column in columns:
            try:
                p_value = adfuller(self.df[column].dropna(), autolag="AIC")[1]
                results[column] = p_value < 0.05
            except Exception as e:
                results[column] = f"Error: {e}"
        return results
<<<<<<< HEAD

>>>>>>> dd52230 (Improvement of the docstring in  data_stream.py)
=======
>>>>>>> f08ccff (Update: improvements to data_stream, ensemble, and exporter modules)
