import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts
import statsmodels.tsa.stattools as ststls
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.robust.scale import mad
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller

# ----------------------- Utilities -----------------------


def deduplicate_history(history):
    """
    Keep only the most recent occurrence of each operation, preserving order.
    """
    seen = set()
    out = []
    for entry in reversed(history):
        op = entry.get("operation")
        if op not in seen:
            out.append(entry)
            seen.add(op)
    return list(reversed(out))


def to_native_types(obj):
    """
    Convert numpy scalars/arrays to Python native types recursively.
    """
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t([to_native_types(v) for v in obj])
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _diagnostics_view(full_history, diagnostics="compact"):
    """
    Format the history for display without losing the full copy stored on the object.
    - 'none'    -> return None (omit)
    - 'compact' -> return deduplicated, small options only
    - 'full'    -> return full (deduplicated) entries
    """
    if diagnostics is None:
        diagnostics = "compact"
    diagnostics = diagnostics.lower()
    if diagnostics == "none":
        return None

    hist = deduplicate_history(full_history)

    if diagnostics == "full":
        return to_native_types(hist)

    # compact: trim obviously large / verbose fields if they ever appear
    compact_hist = []
    for h in hist:
        op = h.get("operation")
        opts = dict(h.get("options", {}))

        # Defensive trims (these keys are never saved by default here,
        # but we keep this in case callers add heavy payloads later).
        for heavy_key in (
            "step_info",
            "common_columns",
            "unique_steps",
            "step_blocks",
            "common_time_grid",
        ):
            if heavy_key in opts:
                opts.pop(heavy_key, None)

        compact_hist.append({"operation": op, "options": opts})
    return to_native_types(compact_hist)


# ----------------------- Core Class -----------------------


class DataStream:
    """
    Time-series analysis with provenance tracking.

    - Full history is always kept internally (self._history).
    - Each method can *display* metadata as 'none' | 'compact' | 'full' via the `diagnostics` arg.
      ('compact' by default). The full internal history remains intact regardless.
    """

    def __init__(self, df: pd.DataFrame, _history=None):
        self.df = df
        self._history = list(_history) if _history is not None else []

    # ---- history helpers ----
    def _add_history(self, operation, options):
        options = {
            k: v
            for k, v in options.items()
            if k not in ("self", "cls", "__class__")
        }
        self._history.append({"operation": operation, "options": options})

        Private helper; not intended for external use.
        """
        options = {
            k: v for k, v in options.items() if k not in ("self", "cls", "__class__")
        }
        self._history.append({"operation": operation, "options": options})

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
        return self.df.head(n)

    def __len__(self):
        return len(self.df)

    def variables(self):
        return self.df.columns

    def _get_columns(self, column_name):
        if column_name is None:
            return [c for c in self.df.columns if c != "time"]
        return (
            [column_name] if isinstance(column_name, str) else list(column_name)
        )

    # ===================== Trimming & Steady-State =====================

    def trim(
        self,
        column_name,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        diagnostics="compact",
    ):
        """
        Trim to steady state using one of: 'std', 'threshold', 'rolling_variance'.
        Returns a new DataStream (may be empty) with full history preserved internally.
        """
        # Stationarity check
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
                f"Column '{column_name}' is not stationary. Trimming requires stationarity."
            )
            new_history.append({"operation": "trim", "options": options})
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

        # Preprocess
        data = self.df[self.df["time"] >= start_time].reset_index(drop=True)
        pos_idx = data.index[data[column_name] > 0]
        if len(pos_idx) > 0:
            non_zero_index = int(pos_idx.min())
            if non_zero_index > 0:
                data = data.loc[non_zero_index:].reset_index(drop=True)

        # Detect steady state
        if method == "std":
            sss = self.find_steady_state_std(
                data, column_name, window_size=batch_size, robust=robust
            )
        elif method == "threshold":
            if threshold is None:
                options["message"] = (
                    "Threshold must be specified for method='threshold'."
                )
                new_history.append({"operation": "trim", "options": options})
                return DataStream(
                    self.df.iloc[0:0].copy(), _history=new_history
                )
            sss = self.find_steady_state_threshold(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        elif method == "rolling_variance":
            threshold = threshold if threshold is not None else 0.1
            sss = self.find_steady_state_rolling_variance(
                data, column_name, window_size=batch_size, threshold=threshold
            )
        elif method == "self_consistent":
            sss = self.find_steady_state_self_consistent(
                data, column_name, window_size=batch_size, robust=robust
            )
        elif method == "iqr":
            thr = 0.05 if threshold is None else float(threshold)
            sss = self.find_steady_state_iqr(
                data, column_name, window_size=batch_size, threshold=thr
            )

        else:
            options["message"] = (
                "Invalid method. Choose 'std', 'threshold', 'rolling_variance', or 'self_consistent'."
            )
            new_history.append({"operation": "trim", "options": options})
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

        options["sss_start"] = sss
        new_history.append({"operation": "trim", "options": options})

        if sss is None:
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

    def trim_sss_start(self, col, workflow):
        """
        Identify and trim the signal to the start of the Statistical Steady State (SSS)

        Parameters
        ----------
        col : str
            The name of the column in `self.df` to analyze for steady state.
        workflow : object
            A configuration/workflow object containing parameters:
            - `_max_lag_frac`: Fraction of data used for autocorrelation lag.
            - `_verbosity`: Integer controlling plot and print output levels.
            - `_autocorr_sig_level`: Significance level for the Z-test on lags.
            - `_decor_multiplier`: Multiplier for the calculated decorrelation length.
            - `_std_dev_frac`: Fraction of standard deviation used for tolerance.
            - `_fudge_fac`: Constant to prevent zero-tolerance in noiseless signals.
            - `_smoothing_window_correction`: Factor to adjust for rolling mean lag.
            - `_final_smoothing_window`: Window size for smoothing the metric curves.

        Returns
        -------
        DataStream
            A new DataStream object containing the DataFrame trimmed to the SSS start.
            Returns an empty DataFrame if no SSS is identified.
        """
        # Get the decorrelation length (in number of points)
        # Note: this approach assumes signal points are spaced equally in time
        n_pts = len(self.df)
        max_lag = int(workflow._max_lag_frac * n_pts)  # max lag for autocorrelation

        acf_vals = ststls.acf(self.df[col].dropna().values, nlags=max_lag)

        # plot the autocorrelation function
        if workflow._verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.stem(range(len(acf_vals)), acf_vals)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Autocorrelation Function")
            plt.grid()
            plt.show()
            plt.close()

        # Use rigorous statistical measure for decorrelation length
        z_critical = sts.norm.ppf(1 - workflow._autocorr_sig_level / 2)
        conf_interval = z_critical / np.sqrt(n_pts)
        significant_lags = np.where(np.abs(acf_vals[1:]) > conf_interval)[0]
        acf_sum = np.sum(np.abs(acf_vals[1:][significant_lags]))
        decor_length = int(np.ceil(1 + 2 * acf_sum))

        # Set smoothing window as multiple of decorrelation length, but not more than max_lag
        decor_index = min(int(workflow._decor_multiplier * decor_length), max_lag)

        if workflow._verbosity > 0:
            print(
                f"stats decorrelation length {decor_length} gives smoothing window of {decor_index} points."
            )

        # Smooth signal with rolling mean over window size based on decorrelation length
        rolling_window = max(3, decor_index)  # at least 3 points in window
        col_smoothed = (
            self.df[col].rolling(window=rolling_window).mean()
        )  # get smoothed column as Series
        col_sm_flld = col_smoothed.bfill()  # fill initial NaNs with first valid value
        # create new DataFrame with time and smoothed flux
        df_smoothed = pd.DataFrame({"time": self.df["time"], col: col_sm_flld})

        # Compute std dev of original signal from current location till end of signal
        std_dev_till_end = np.empty((n_pts,), dtype=float)
        for i in range(n_pts):
            std_dev_till_end[i] = np.std(self.df[col].iloc[i:])
        # turn this into a pandas series with same index as col_smoothed
        std_dev_till_end_series = pd.Series(std_dev_till_end, index=self.df.index)
        # Smooth this std dev to avoid it going to zero at end of signal
        std_dev_smoothed = std_dev_till_end_series.rolling(
            window=workflow._final_smoothing_window
        ).mean()
        # Fill initial NaNs with the first valid smoothed std dev value
        std_dev_sm_flld = std_dev_smoothed.bfill()

        # create new DataFrame with time and std dev till end of signal
        df_std_dev = pd.DataFrame(
            {"time": self.df["time"], col + "_std_till_end": std_dev_sm_flld}
        )

        # start time of smoothed signal
        smoothed_start_time = df_smoothed["time"].iloc[rolling_window - 1]

        # plot smoothed signal and related quantities
        if workflow._verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.df["time"],
                self.df[col],
                label="Original Signal",
                alpha=0.5,
            )
            plt.plot(
                df_smoothed["time"],
                df_smoothed[col],
                label="Smoothed Signal",
                color="orange",
            )
            plt.plot(
                df_std_dev["time"],
                df_std_dev[col + "_std_till_end"],
                label="Smoothed Std Dev Till End",
                color="green",
            )
            plt.axvline(
                x=smoothed_start_time,
                color="g",
                linestyle="--",
                label="First smoothed point",
            )
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.title("Original and Smoothed Signal")
            plt.legend()
            plt.grid()
            plt.show()
            plt.close()

        if workflow._verbosity > 0:
            print("Getting start of SSS based on smoothed signal:")

        # Get start of SSS based on where the value of the flux in the smoothed signal
        # is close to the mean of the remaining signal.

        # At each location, compute the mean of the remaining smoothed signal
        n_pts_smoothed = len(df_smoothed)
        mean_vals = np.empty((n_pts_smoothed,), dtype=float)

        for i in range(n_pts_smoothed):
            mean_vals[i] = np.mean(df_smoothed[col].iloc[i:])

        # Check where the current value of the smoothed signal is within tol_fac of the mean of the remaining signal
        deviation_arr = np.abs(df_smoothed[col] - mean_vals)

        # smooth this so the deviation does not go to zero at end of signal by construction
        # turn this into a pandas series with same index as col_smoothed
        deviation_series = pd.Series(deviation_arr, index=self.df.index)
        # Smooth this std dev to avoid it going to zero at end of signal
        deviation_smoothed = deviation_series.rolling(
            window=workflow._final_smoothing_window
        ).mean()
        # Fill initial NaNs with the first valid smoothed std dev value
        deviation_sm_flld = deviation_smoothed.bfill()
        # Build a dataframe for the deviation
        deviation = pd.DataFrame(
            {"time": self.df["time"], col + "_deviation": deviation_sm_flld}
        )

        # Compute tolerance on variation in the mean of the smoothed signal as
        # stdv_frac * (std dev till end + a fudge factor * mean value at start of smoothed signal)
        # fudge factor is for in case there is no noise (and to guard against the tolerance
        # factor going to zero when std dev gets very small at end of signal)
        tol_fac = workflow._std_dev_frac * (
            df_std_dev[col + "_std_till_end"] + workflow._fudge_fac * abs(mean_vals[0])
        )
        tolerance = tol_fac * np.abs(mean_vals)

        within_tolerance_all = deviation[col + "_deviation"] <= tolerance
        # Only consider points after the smoothed signal has started
        within_tolerance = within_tolerance_all & (
            df_smoothed["time"] >= smoothed_start_time
        )
        # First index where we are within tolerance
        sss_index = np.where(within_tolerance)[0]

        # See if there is a segment where ALL remaining points are within tolerance
        crit_met_index = None
        if len(sss_index) > 0:
            # find the segment where ALL remaining points are within tolerance
            for idx in sss_index:
                if np.all(within_tolerance[idx:]):
                    crit_met_index = idx
                    break

        if crit_met_index is not None:  # We have a SSS segment
            # Time where criterion has been met
            criterion_time = df_smoothed["time"].iloc[crit_met_index]
            # Take into account that the signal at the point where the criterion has been met is a result
            # of averaging over the rolling window. So set the start of SSS near the start of the rolling window
            # but not all the way at the beginning of the rolling window as there is usually still some transient.
            true_sss_start_index = max(
                0,
                int(
                    crit_met_index
                    - workflow._smoothing_window_correction * rolling_window
                ),
            )
            sss_start_time = df_smoothed["time"].iloc[true_sss_start_index]

            if workflow._verbosity > 0:
                print(f"Index where criterion is met: {crit_met_index}")
                print(f"Rolling window: {rolling_window}")
                print(f"time where criterion is met: {criterion_time}")
                print(
                    f"time at start of SSS (adjusted for rolling window): {sss_start_time}"
                )

            # Plot deviation and tolerance vs. time
            if workflow._verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    df_smoothed["time"],
                    deviation[col + "_deviation"],
                    label="Deviation",
                    color="blue",
                )
                plt.plot(
                    df_smoothed["time"],
                    tolerance,
                    label="Tolerance",
                    color="orange",
                )
                plt.axvline(
                    x=criterion_time,
                    color="g",
                    linestyle="--",
                    label="Small Change Criterion Met",
                )
                plt.axvline(
                    x=sss_start_time, color="r", linestyle="--", label="Start SSS"
                )
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title("Deviation and Tolerance vs. Time")
                plt.legend()
                plt.grid()
                plt.show()
                plt.close()

            # Trim the original data frame to start at this location minus the smoothing window
            trimmed_df = self.df[self.df["time"] >= sss_start_time]
            # Reset the index so it starts at 0
            trimmed_df = trimmed_df.reset_index(drop=True)
            # Create new data stream from trimmed data frame
            trimmed_stream = DataStream(trimmed_df)

        else:
            if workflow._verbosity > 0:
                print("No SSS found based on behavior of mean of smoothed signal.")
            trimmed_stream = pd.DataFrame(
                columns=["time", "flux"]
            )  # Create empty DataFrame with same columns as original

            # Plot deviation and tolerance vs. time
            if workflow._verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    df_smoothed["time"],
                    deviation[col + "_deviation"],
                    label="Deviation",
                    color="blue",
                )
                plt.plot(
                    df_smoothed["time"],
                    tolerance,
                    label="Tolerance",
                    color="orange",
                )
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title("Deviation and Tolerance vs. Time")
                plt.legend()
                plt.grid()
                plt.show()
                plt.close()

        return trimmed_stream

    @staticmethod
    def find_steady_state_std_old(
        data, column_name, window_size=10, robust=True
    ):
        time_vals = data["time"].values
        x = data[column_name].values
        for i in range(len(x) - window_size + 1):
            rem = x[i:]
            if robust:
                center = np.median(rem)
                scale = mad(rem)
                if scale == 0:
                    scale = np.std(rem)  # fallback
                if scale == 0:
                    continue  # completely constant or degenerate segment
            else:
                center = np.mean(rem)
                scale = np.std(rem)
            within_1 = np.mean(np.abs(rem - center) <= 1 * scale)
            within_2 = np.mean(np.abs(rem - center) <= 2 * scale)
            within_3 = np.mean(np.abs(rem - center) <= 3 * scale)
            if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
                return time_vals[i]
        return None

    @staticmethod
    def find_steady_state_rolling_variance(
        data, column_name, window_size=50, threshold=0.1
    ):
        ts = data[["time", column_name]].dropna()
        rv = ts[column_name].rolling(window=window_size).std()
        thr = rv.mean() * threshold
        idx = np.where(rv < thr)[0]
        if len(idx) > 0:
            return ts["time"].iloc[idx[0]]
        return None

    @staticmethod
    def normalize_data(df):
        scaler = MinMaxScaler()
        if df.shape[1] > 1:
            df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
        return df

    @staticmethod
    def find_steady_state_threshold(data, column_name, window_size, threshold):
        d = DataStream.normalize_data(data.copy())
        if len(d) < window_size:
            return None
        rs = d[column_name].rolling(window=window_size).std()
        common_idx = d.index.intersection(rs.index)
        steady = d.loc[common_idx, "time"][rs.loc[common_idx] < threshold]
        if not steady.empty:
            return steady.iloc[0]
        return None

    @staticmethod
    def find_steady_state_self_consistent(
        data,
        column_name,
        window_size=10,
        robust=True,
    ):
        """
        Upgrade A self-consistency steady-state detection.

        Parameters match the other steady-state helpers:
          - data: pre-filtered dataframe (already start_time + first-positive handled in trim)
          - column_name: series to analyze
          - window_size: W (block length)
          - robust: use median+MAD (scaled) vs mean+std

        Returns
        -------
        float | None
            time value where steady state begins, or None if not found.
        """
        if data is None or data.empty:
            return None
        if "time" not in data.columns or column_name not in data.columns:
            return None

        t = data["time"].to_numpy(dtype=float)
        x = data[column_name].to_numpy(dtype=float)

        W = int(window_size)
        if W < 1:
            return None

        n = x.size
        if n < 2 * W:
            # need at least two full blocks to compare
            return None

        # 5% tolerances
        rel_tol_mu = 0.1
        rel_tol_sigma = 0.05
        eps_floor = 1e-12  # avoids 0 tolerance when mu or sigma ~ 0

        def block_mu_sigma(arr: np.ndarray):
            arr = np.asarray(arr, dtype=float)
            if arr.size == 0:
                return np.nan, np.nan

            if robust:
                mu = float(np.median(arr))
                s = float(mad(arr, center=mu))
                # MAD -> sigma consistency for normal data
                s = 1.4826 * s
                if not np.isfinite(s) or s <= 0:
                    s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            else:
                mu = float(np.mean(arr))
                s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

            if not np.isfinite(mu):
                mu = np.nan
            if not np.isfinite(s):
                s = np.nan
            return mu, s

        best_time = None

        # offsets i = 0..W-1
        for i in range(0, W):
            m = (n - i) // W  # number of full blocks
            if m < 2:
                continue

            mus = np.empty(m, dtype=float)
            sigs = np.empty(m, dtype=float)
            starts = np.empty(m, dtype=int)

            for k in range(m):
                a = i + k * W
                b = a + W
                mus[k], sigs[k] = block_mu_sigma(x[a:b])
                starts[k] = a

            # find first k0 where ALL consecutive pairs from k0..end pass
            for k0 in range(0, m - 1):
                ok = True
                for k in range(k0, m - 1):
                    muA, muB = mus[k], mus[k + 1]
                    sA, sB = sigs[k], sigs[k + 1]

                    if not (
                        np.isfinite(muA)
                        and np.isfinite(muB)
                        and np.isfinite(sA)
                        and np.isfinite(sB)
                    ):
                        ok = False
                        break

                    eps_mu = max(
                        eps_floor, rel_tol_mu * max(eps_floor, abs(muB))
                    )
                    eps_s = max(
                        eps_floor, rel_tol_sigma * max(eps_floor, abs(sB))
                    )

                    if abs(muA - muB) > eps_mu:
                        ok = False
                        break
                    if abs(sA - sB) > eps_s:
                        ok = False
                        break

                if ok:
                    candidate_time = float(t[int(starts[k0])])
                    if best_time is None or candidate_time < best_time:
                        best_time = candidate_time
                    break  # earliest k0 for this offset

        return best_time

    @staticmethod
    def find_steady_state_std(data, column_name, window_size=10, robust=True):
        time_vals = data["time"].values
        x = data[column_name].values

        # thresholds for Upgrade 2 (robust, distribution-agnostic)
        z1, p1 = 2.5, 0.95
        z2, p2 = 4.0, 0.99
        eps_floor = 1e-12

        for i in range(len(x) - window_size + 1):
            rem = x[i:]

            if robust:
                # --- ROBUST PATH ---
                med = np.median(rem)
                s = 1.4826 * mad(rem)

                if not np.isfinite(s) or s <= eps_floor:
                    s = np.std(rem, ddof=1) if rem.size > 1 else 0.0
                    if s <= eps_floor:
                        continue

                z = np.abs(rem - med) / s
                if (np.mean(z <= z1) >= p1) and (np.mean(z <= z2) >= p2):
                    return time_vals[i]

            else:
                # --- CLASSICAL NORMAL PATH ---
                center = np.mean(rem)
                scale = np.std(rem)

                if scale <= eps_floor:
                    continue

                within_1 = np.mean(np.abs(rem - center) <= 1 * scale)
                within_2 = np.mean(np.abs(rem - center) <= 2 * scale)
                within_3 = np.mean(np.abs(rem - center) <= 3 * scale)

                if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
                    return time_vals[i]

        return None

    @staticmethod
    def find_steady_state_iqr(
        data, column_name, window_size=10, threshold=0.05
    ):
        """
        IQR-based steady state:
        IQR(rem) <= threshold * max(eps, |median(rem)|)
        threshold default 0.05 => 5%
        """
        time_vals = data["time"].values
        x = data[column_name].values
        eps_floor = 1e-12

        for i in range(len(x) - window_size + 1):
            rem = x[i:]
            q25, q75 = np.percentile(rem, [25, 75])
            iqr = q75 - q25
            med = np.median(rem)
            scale_ref = max(eps_floor, abs(med))
            if iqr <= (threshold * scale_ref):
                return time_vals[i]

        return None

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

    # ===================== ESS (Default: Geyer) =====================

    def effective_sample_size(
        self,
        column_names=None,
        method="geyer",
        alpha=0.05,
        diagnostics="compact",
    ):
        """
        ESS on the raw series (not block means).
          - method='geyer' (alias: 'geyser'): positive-pair truncation (conservative).
          - method='significance': include |rho_k| only for significant lags.
        Returns {'results': {col: int_ESS or None/message}, 'metadata': ...}
        """
        method_str = (method or "geyer").lower()
        if method_str == "geyser":
            method_str = "geyer"

        self._add_history(
            "effective_sample_size",
            {
                "column_names": column_names,
                "method": method_str,
                "alpha": alpha,
            },
        )

        if column_names is None:
            cols = [c for c in self.df.columns if c != "time"]
        elif isinstance(column_names, str):
            cols = [column_names]
        else:
            cols = list(column_names)

        out = {}
        for col in cols:
            if col not in self.df.columns:
                out[col] = {"message": f"Column '{col}' not found."}
                continue
            x = self.df[col].dropna().values
            if x.size == 0:
                out[col] = {
                    "effective_sample_size": None,
                    "message": "No data.",
                }
                continue

            n = len(x)
            nlags = max(1, n // 4)
            # Use fft=False for small/mid series stability
            r = acf(x, nlags=nlags, fft=False)

            if method_str == "geyer":
                # Positive-pair rule
                s, t = 0.0, 1
                while t + 1 < len(r):
                    pair_sum = r[t] + r[t + 1]
                    if pair_sum < 0:
                        break
                    s += pair_sum
                    t += 2
                ess = n / (1 + 2 * s)
            else:
                # significance-filtered absolute sum
                zcrit = norm.ppf(1 - alpha / 2)
                band = zcrit / np.sqrt(n)
                sig = np.where(np.abs(r[1:]) > band)[0]
                acf_sum = np.sum(np.abs(r[1:][sig])) if sig.size else 0.0
                ess = n / (1 + 2 * acf_sum)

            ess = float(max(1.0, min(ess, n)))
            out[col] = {"effective_sample_size": int(math.ceil(ess))}
            # out[col] = int(math.ceil(ess))

        metadata = _diagnostics_view(self._history, diagnostics=diagnostics)
        return {"results": to_native_types(out), "metadata": metadata}

    def get_block_effective_n(
        self,
        column_name,
        method="non-overlapping",
        window_size=None,
        diagnostics="none",
    ):
        """
        Compute effective_n (ESS) *on the block means* using Geyer positive-pair truncation.
        Returns a small dict you can reuse anywhere:
        {'effective_n': float, 'window_size': int, 'n_blocks': int}

        Notes
        -----
        - Uses self._estimate_window(...) when window_size is None.
        - Uses self._process_column(...) to form block means.
        - Always caps ESS at >= 1.0.
        """
        series = self.df[column_name].dropna()
        if series.empty:
            return {
                "effective_n": float("nan"),
                "window_size": window_size,
                "n_blocks": 0,
            }

        est_win = self._estimate_window(column_name, series, window_size)
        proc = self._process_column(series, est_win, method)
        bm = proc.values
        n_blocks = len(bm)

        if n_blocks >= 3:
            r = acf(bm, nlags=max(1, n_blocks // 4), fft=False)
            s, t = 0.0, 1
            while t + 1 < len(r):
                pair_sum = r[t] + r[t + 1]
                if pair_sum < 0:
                    break
                s += pair_sum
                t += 2
            ess_blocks = max(1.0, n_blocks / (1 + 2 * s))
        else:
            ess_blocks = 1.0

        # Keep full history internally (compact fields only)
        self._add_history(
            "get_block_effective_n",
            {
                "column_name": column_name,
                "method": method,
                "window_size": window_size,
                "est_win": int(est_win),
                "n_blocks": int(n_blocks),
                "effective_n": float(ess_blocks),
            },
        )

        if diagnostics != "none":
            _ = self.get_metadata(
                diagnostics=diagnostics
            )  # just to keep behavior consistent

        return {
            "effective_n": float(ess_blocks),
            "window_size": int(est_win),
            "n_blocks": int(n_blocks),
        }

    # ===================== Tau_int + Ljung-Box window autotune =====================

    @staticmethod
    def _tau_int_geyer_from_acf(rho: np.ndarray) -> float:
        """
        Estimate integrated autocorrelation time tau_int using Geyer positive-pair truncation
        on an ACF array rho where rho[0]=1.
        """
        if rho is None or len(rho) < 2:
            return 1.0

        s, t = 0.0, 1
        while t + 1 < len(rho):
            pair_sum = rho[t] + rho[t + 1]
            if pair_sum < 0:
                break
            s += pair_sum
            t += 2
        tau = 1.0 + 2.0 * s
        return float(max(1.0, tau))

    def _estimate_tau_int(self, series: pd.Series) -> float:
        """
        Estimate tau_int directly from the raw series ACF using Geyer positive-pair truncation.
        This is more 'standard' than backing out tau_int from ESS via n/ESS, but consistent.
        """
        x = np.asarray(series.dropna().values, dtype=float)
        n = x.size
        if n < 3:
            return 1.0

        # nlags choice: common rule is n//4, but cap to keep it stable
        nlags = max(1, min(n // 4, 2000))
        r = acf(x, nlags=nlags, fft=False)
        return self._tau_int_geyer_from_acf(r)

    def _ljung_box_pass(
        self,
        block_means: np.ndarray,
        alpha: float = 0.05,
        lag_set=(5, 10),
    ) -> tuple[bool, dict]:
        """
        Ljung–Box test on block means. Tests multiple lags (capped by n_blocks-1).
        Pass means pvalue > alpha for ALL tested lags.
        Returns (passed, details_dict).
        """
        bm = np.asarray(block_means, dtype=float)
        n_blocks = bm.size
        if n_blocks < 2:
            return False, {"n_blocks": int(n_blocks), "lags": [], "pvalues": []}

        tested_lags = []
        pvalues = []

        for lag in lag_set:
            L = int(min(lag, n_blocks - 1))
            if L < 1:
                continue
            lb = acorr_ljungbox(bm, lags=[L], return_df=True)
            p = float(lb["lb_pvalue"].iloc[0])
            tested_lags.append(L)
            pvalues.append(p)

        if not tested_lags:
            return False, {"n_blocks": int(n_blocks), "lags": [], "pvalues": []}

        passed = all(p > alpha for p in pvalues)
        return passed, {
            "n_blocks": int(n_blocks),
            "lags": tested_lags,
            "pvalues": pvalues,
        }

    def _autotune_window_size(
        self,
        col: str,
        series: pd.Series,
        method: str = "non-overlapping",
        alpha: float = 0.05,
        lag_set=(5, 10),
        B_min: int = 15,
        c0: float = 2.0,
        max_iter: int = 12,
        w_min: int = 5,
        diagnostics: bool = False,
    ):
        """
        Choose block window size using:
          1) tau_int estimate (Geyer on raw ACF)
          2) w0 = ceil(c0 * tau_int)
          3) Ljung–Box test on block means; if fail -> increase w via max(1.5*w, w + tau_int)
          4) enforce minimum block count B_min when possible
          5) fallback to best p-value if never passes

        Returns
        -------
        (chosen_w:int, info:dict)
        """
        x = series.dropna()
        n = int(x.size)
        if n < 2:
            return w_min, {
                "status": "too_few_blocks",
                "tau_int": 1.0,
                "chosen_w": int(w_min),
                "passed": False,
                "lb": {"lags": [], "pvalues": []},
                "note": "Too few samples.",
            }

        tau = self._estimate_tau_int(x)
        w = int(max(w_min, math.ceil(c0 * tau)))

        # Ensure we *try* to keep at least B_min blocks if possible:
        # i.e., w <= n//B_min. If n//B_min < w_min, we can't enforce.
        w_cap_for_blocks = (n // B_min) if (B_min and n // B_min >= 1) else n
        if w_cap_for_blocks >= w_min:
            w = min(w, w_cap_for_blocks)

        best = {
            "w": w,
            "passed": False,
            "p_min": -np.inf,
            "details": None,
            "lb": {"lags": [], "pvalues": []},
        }

        for it in range(int(max_iter)):
            proc = self._process_column(
                x, estimated_window=w, method="non-overlapping"
            )
            bm = np.asarray(proc.values, dtype=float)
            n_blocks = int(bm.size)

            # If we can’t even form 2 blocks, stop.
            if n_blocks < 2:
                break

            passed, det = self._ljung_box_pass(bm, alpha=alpha, lag_set=lag_set)
            # p_min = min(det["pvalues"]) if det.get("pvalues") else -np.inf

            # choose a single score for "best" even if multiple pvalues
            p_score = min(det["pvalues"]) if det.get("pvalues") else -np.inf
            if p_score > best["p_min"]:
                best = {
                    "w": int(w),
                    "passed": bool(passed),
                    "p_min": float(p_score),
                    "details": det,
                    "lb": det,
                }

            if passed:
                return int(w), {
                    "status": "independent",
                    "tau_int": float(tau),
                    "chosen_w": int(w),
                    "passed": True,
                    "iter": int(it),
                    "lb": det,
                    "note": None,
                }

            # If we already hit a cap that preserves blocks, and still fail, we may need to
            # relax the cap and accept fewer blocks (more conservative) to reach independence.
            # We'll allow w to grow beyond w_cap_for_blocks after the first failure.
            w_next = int(w + 1)
            # w_next = int(math.ceil(max(1.5 * w, w + tau)))

            # avoid infinite loop if no change
            if w_next <= w:
                break
            w = w_next

            # If window gets so big we have <2 blocks, stop
            if n // w < 2:
                break

        # If we never passed but have a best window with >=2 blocks, use it
        if best["p_min"] > -np.inf:
            return int(best["w"]), {
                "status": "best_p",
                "tau_int": float(tau),
                "chosen_w": int(best["w"]),
                "passed": False,
                "iter": int(max_iter),
                "lb": best["lb"],
                "note": "Did not reach Ljung–Box pass; using best observed p-value window.",
            }
        # Otherwise: too few blocks to test
        return int(w), {
            "status": "too_few_blocks",
            "tau_int": float(tau),
            "chosen_w": int(w),
            "passed": False,
            "lb": {"lags": [], "pvalues": []},
            "note": "Window growth left <2 blocks; cannot test independence.",
        }
        #    # fallback: best observed window (even if it didn't pass)
        #    note = "Did not pass Ljung–Box within max_iter; using best observed p-value."
        # return int(best["w"]), {
        #    "tau_int": float(tau),
        #    "chosen_w": int(best["w"]),
        #    "passed": bool(best["passed"]),
        #    "iter": int(max_iter),
        #    "lb": best["details"],
        #    "note": note,
        # }

    # ===================== Block/Window Stats =====================

    def _estimate_window(self, col, column_data, window_size):
        """
        If window_size is None:
          - NEW: use tau_int + Ljung–Box autotune on block means
        Else:
          - respect the user-provided window_size

        NOTE: This keeps the same signature so the rest of your code stays intact.
        """
        if window_size is not None:
            return int(window_size)

        # --- NEW: statistically standard autotune ---
        w, info = self._autotune_window_size(
            col=col,
            series=column_data,
            method="non-overlapping",  # match your default pipeline
            alpha=0.05,
            lag_set=(5, 10),
            B_min=15,
            c0=2.0,
            max_iter=20,
            w_min=5,
        )

        # Log what happened (compact, light)
        self._add_history(
            "estimate_window_tau_lb",
            {
                "column_name": col,
                "chosen_w": int(w),
                "tau_int": float(info.get("tau_int", np.nan)),
                "status": info.get("status"),
                "passed": bool(info.get("passed", False)),
                "lb_lags": info.get("lb", {}).get("lags", None),
                "lb_pvalues": info.get("lb", {}).get("pvalues", None),
                "note": info.get("note", None),
            },
        )
        return int(w)

    def _estimate_window_old(self, col, column_data, window_size):
        """
        If window_size is None, derive a coarse block length from ESS (geyer).
        Ensure a minimum window of 5.
        """
        if window_size is None:
            ess_info = self.effective_sample_size(
                column_names=col, method="geyer", diagnostics="none"
            )
            ess_val = 10
            try:
                ess_val = int(max(1, ess_info["results"].get(col, 10)))
            except Exception:
                pass
            return max(5, len(column_data) // ess_val)
        return int(window_size)

    def _process_column(self, column_data, estimated_window, method):
        if method == "sliding":
            return (
                column_data.rolling(window=int(estimated_window))
                .mean()
                .dropna()
            )

        elif method == "non-overlapping":
            w = int(max(1, estimated_window))
            x = np.asarray(column_data.values, dtype=float)
            n = x.size
            n_blocks = n // w
            if n_blocks < 1:
                return pd.Series([], dtype=float)

            # drop remainder
            x2 = x[: n_blocks * w].reshape(n_blocks, w)
            block_means = x2.mean(axis=1)

            # center index per block (optional)
            idx = (np.arange(n_blocks) * w) + (w // 2)
            return pd.Series(block_means, index=idx)

        else:
            raise ValueError(
                "Invalid method. Choose 'sliding' or 'non-overlapping'."
            )

    def _process_column_old(self, column_data, estimated_window, method):
        if method == "sliding":
            return column_data.rolling(window=estimated_window).mean().dropna()
        elif method == "non-overlapping":
            step = max(1, estimated_window)
            window_means = [
                np.mean(column_data[i : i + estimated_window])
                for i in range(0, len(column_data) - estimated_window + 1, step)
            ]
            idx = np.arange(
                estimated_window // 2,
                len(window_means) * step + estimated_window // 2,
                step,
            )
            return pd.Series(window_means, index=idx)
        else:
            raise ValueError(
                "Invalid method. Choose 'sliding' or 'non-overlapping'."
            )

    def _mean(
        self, column_name=None, method="non-overlapping", window_size=None
    ):
        results = {}
        for col in self._get_columns(column_name):
            data = self.df[col].dropna()
            if data.empty:
                results[col] = {
                    "mean": np.nan,
                    "window_size": np.nan,
                    "error": f"No data for '{col}'",
                }
                continue
            est_win = self._estimate_window(col, data, window_size)
            proc = self._process_column(data, est_win, method)
            results[col] = {
                "mean": float(np.mean(proc)),
                "window_size": int(est_win),
            }
        return results

    def _mean_uncertainty(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        """
        SEM of block/sliding means.
        Uses Geyer-derived effective_n from get_block_effective_n() on the *block means*.
        """
        results = {}
        for col in self._get_columns(column_name):
            series = self.df[col].dropna()
            if series.empty:
                results[col] = {
                    "mean_uncertainty": np.nan,
                    "window_size": np.nan,
                    "error": f"No data for '{col}'",
                }
                continue

            #    Pull effective_n and chosen window from the helper
            info = self.get_block_effective_n(
                col, method=method, window_size=window_size, diagnostics="none"
            )
            est_win = info["window_size"]
            eff_n = (
                info["effective_n"] if np.isfinite(info["effective_n"]) else 1.0
            )

            #    Recompute block means (cheap) to get their std with ddof
            proc = self._process_column(series, est_win, method)
            block_means = proc.values

            unc = float(np.std(block_means, ddof=ddof) / np.sqrt(eff_n))
            results[col] = {
                "mean_uncertainty": unc,
                "window_size": int(est_win),
            }
        return results

    ## Added variance calculation of block means
    def _variance(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        """
        Variance of block/sliding means.

        - Uses the same block construction and window selection as _mean_uncertainty:
          * get_block_effective_n(...) to choose window_size and compute ESS on block means.
          * _process_column(...) to form block means.
        - Returns the variance of the block means (not the raw sample variance), plus
          the block-level effective_n and window_size.
        """
        results = {}
        for col in self._get_columns(column_name):
            series = self.df[col].dropna()
            if series.empty:
                results[col] = {
                    "variance": np.nan,
                    "window_size": np.nan,
                    "effective_n_blocks": np.nan,
                    "error": f"No data for '{col}'",
                }
                continue

            # Pull effective_n and chosen window from the helper
            info = self.get_block_effective_n(
                col, method=method, window_size=window_size, diagnostics="none"
            )
            est_win = info["window_size"]
            eff_n_blocks = info["effective_n"]

            # Recompute block means (cheap) to get their variance
            proc = self._process_column(series, est_win, method)
            block_means = proc.values

            if block_means.size < 2:
                var_val = np.nan
            else:
                var_val = float(np.var(block_means, ddof=ddof))

            results[col] = {
                "variance": var_val,
                "window_size": int(est_win),
                "effective_n_blocks": float(eff_n_blocks),
            }
        return results

    def _short_term_counts(
        self,
        column_name=None,
        method="non-overlapping",
        window_size=None,
    ):
        """
        Count the number of short-term averages (block means) used
        in the block statistics.

        This uses the same window selection and block construction
        as _mean_uncertainty and _variance, via get_block_effective_n().

        Returns
        -------
        dict
            {col: {"n_short_averages": int, "window_size": int, "n_blocks": int}}
        """
        results = {}
        for col in self._get_columns(column_name):
            series = self.df[col].dropna()
            if series.empty:
                results[col] = {
                    "n_short_averages": np.nan,
                    "window_size": np.nan,
                    "n_blocks": 0,
                    "error": f"No data for '{col}'",
                }
                continue

            est_win = self._estimate_window(col, series, window_size)

            # If the window is longer than the data, _process_column will produce nothing.
            if len(series) < est_win:
                results[col] = {
                    "n_short_averages": 0,
                    "window_size": int(est_win),
                    "n_blocks": 0,
                    "error": f"Not enough data for window_size={est_win}",
                }
                continue

            proc = self._process_column(series, est_win, method)
            n_blocks = int(len(proc))

            results[col] = {
                "n_short_averages": n_blocks,
                "window_size": int(est_win),
                "n_blocks": n_blocks,
            }
        return results

    def _mean_uncertainty_old(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        """
        SEM of block/sliding means. Uses Geyer ESS on the *block means* (conservative).
        """
        results = {}
        for col in self._get_columns(column_name):
            data = self.df[col].dropna()
            if data.empty:
                results[col] = {
                    "mean_uncertainty": np.nan,
                    "window_size": np.nan,
                    "error": f"No data for '{col}'",
                }
                continue

            est_win = self._estimate_window(col, data, window_size)
            proc = self._process_column(data, est_win, method)
            block_means = proc.values

            if len(block_means) >= 3:
                r = acf(
                    block_means, nlags=max(1, len(block_means) // 4), fft=False
                )
                s, t = 0.0, 1
                while t + 1 < len(r):
                    pair_sum = r[t] + r[t + 1]
                    if pair_sum < 0:
                        break
                    s += pair_sum
                    t += 2
                ess_blocks = max(1.0, len(block_means) / (1 + 2 * s))
            else:
                ess_blocks = 1.0

            effective_n = ess_blocks
            unc = float(np.std(block_means, ddof=ddof) / np.sqrt(effective_n))
            results[col] = {
                "mean_uncertainty": unc,
                "window_size": int(est_win),
            }
        return results

    def _confidence_interval(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        results = {}
        mean_res = self._mean(
            column_name, method=method, window_size=window_size
        )
        unc_res = self._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        for col in self._get_columns(column_name):
            mu = mean_res.get(col, {}).get("mean", np.nan)
            se = unc_res.get(col, {}).get("mean_uncertainty", np.nan)
            if np.isnan(mu) or np.isnan(se):
                results[col] = {
                    "confidence_interval": (np.nan, np.nan),
                    "window_size": mean_res.get(col, {}).get(
                        "window_size", np.nan
                    ),
                    "error": mean_res.get(col, {}).get("error", "")
                    or unc_res.get(col, {}).get("error", ""),
                }
            else:
                ci = (float(mu - 1.96 * se), float(mu + 1.96 * se))
                results[col] = {
                    "confidence_interval": ci,
                    "window_size": mean_res[col]["window_size"],
                }
        return results

    # ===================== Public Stats API =====================

    def compute_statistics_old(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        diagnostics="compact",
    ):
        """
        For each column: mean, SEM (Geyer on blocks), 95% CI, ±1*SEM band, ESS (raw series),
        window size, and an independence warning (Ljung-Box).

        For each column:
          - mean (block-based)
          - SEM (mean_uncertainty; Geyer on blocks)
          - variance of block means
          - 95% CI
          - ±1*SEM band
          - ESS on raw series
          - block-level ESS (from get_block_effective_n)
          - number of short-term averages (block means)
          - window size
          - independence warning (Ljung-Box on block means).
        """
        stats = {}
        cols = self._get_columns(column_name)

        mean_res = self._mean(
            column_name, method=method, window_size=window_size
        )
        mu_res = self._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ci_res = self._confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        ess_res = self.effective_sample_size(
            column_names=column_name, method="geyer", diagnostics="none"
        )
        # NEW: variance + block-level ESS
        var_res = self._variance(
            column_name, ddof=ddof, method=method, window_size=window_size
        )
        # NEW: number of short-term averages (block means)
        short_res = self._short_term_counts(
            column_name, method=method, window_size=window_size
        )

        for col in cols:
            # Independence test per column
            try:
                ind = self.test_block_independence(
                    col,
                    method=method,
                    window_size=window_size,
                    verbose=False,
                    diagnostics="none",
                )
                independent = ind.get("independent", None)
            except Exception:
                independent = None

            warn = None
            if independent is False:
                warn = "Short-term averages failed independence (autocorrelation detected). Use stats with caution."
            elif independent is None:
                warn = "Block independence undetermined (insufficient data). Use stats with caution."

            data = self.df[col].dropna()
            if data.empty:
                stats[col] = {"error": f"No data for '{col}'"}
                continue

            mu = mean_res[col]["mean"]
            se = mu_res[col]["mean_uncertainty"]
            ci = ci_res[col]["confidence_interval"]
            est_win = mean_res[col]["window_size"]
            ess_val = ess_res["results"].get(col, 10)

            # NEW: variance of block means and block-level effective_n
            var_val = var_res.get(col, {}).get("variance", np.nan)
            block_ess = var_res.get(col, {}).get("effective_n_blocks", np.nan)

            # NEW: number of short-term averages (block means)
            n_short = short_res.get(col, {}).get("n_short_averages", np.nan)

            stats[col] = {
                "mean": mu,
                "mean_uncertainty": se,
                "variance": var_val,  # NEW
                "confidence_interval": ci,
                "pm_std": (mu - se, mu + se),
                "effective_sample_size": ess_val,  # raw-series ESS (existing)
                "block_effective_n": block_ess,  # NEW: ESS on block means
                "n_short_averages": n_short,  # NEW: count of block means
                "window_size": est_win,
            }
            if warn:
                stats[col]["warning"] = warn

        # Append operation and return metadata view
        op_options = dict(
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
        )
        self._add_history("compute_statistics", op_options)
        stats["metadata"] = _diagnostics_view(
            self._history, diagnostics=diagnostics
        )
        return to_native_types(stats)

    def compute_statistics(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        diagnostics="compact",
    ):
        """
        Fast version: compute block means ONCE per column and reuse everywhere.
        """
        stats = {}
        cols = self._get_columns(column_name)

        # Raw-series ESS (keep as-is; computed once for all cols)
        ess_res = self.effective_sample_size(
            column_names=column_name, method="geyer", diagnostics="none"
        )

        for col in cols:
            series = self.df[col].dropna()
            if series.empty:
                stats[col] = {"error": f"No data for '{col}'"}
                continue

            # window selection with enforced independence attempt
            if window_size is None:
                w, info = self._autotune_window_size(
                    col=col,
                    series=series,
                    method=method,
                    alpha=0.05,
                    lag_set=(5, 10),
                    c0=2.0,
                    max_iter=20,
                    w_min=5,
                )
                status = info.get("status", "best_p")
                lb = info.get("lb", {"lags": [], "pvalues": []})

                self._add_history(
                    "estimate_window_tau_lb",
                    {
                        "column_name": col,
                        "chosen_w": int(w),
                        "tau_int": float(info.get("tau_int", np.nan)),
                        "status": status,
                        "passed": bool(info.get("passed", False)),
                        "lb_lags": lb.get("lags", []),
                        "lb_pvalues": lb.get("pvalues", []),
                        "note": info.get("note", None),
                    },
                )
            else:
                w = int(window_size)
                status = "user_window"
                lb = {"lags": [], "pvalues": []}

            # 2) Block/sliding means ONCE
            proc = self._process_column(
                series, estimated_window=w, method=method
            )
            block_means = np.asarray(proc.values, dtype=float)
            n_blocks = int(block_means.size)

            if n_blocks < 1:
                stats[col] = {
                    "error": f"No block means produced (window_size={w}).",
                    "window_size": int(w),
                    "n_short_averages": 0,
                    "independence_status": status,
                }
                continue

            # 3) Block statistics from the SAME block_means
            mu = float(np.mean(block_means))

            var_val = (
                float(np.var(block_means, ddof=ddof))
                if n_blocks >= 2
                else np.nan
            )
            sd = (
                float(np.std(block_means, ddof=ddof))
                if n_blocks >= 2
                else np.nan
            )

            # ESS on block means (used for the last fallback path)
            ess_blocks = self._geyer_ess_on_blocks(block_means)

            # Decide SE rule
            warning = None
            # Decide how to compute SE from block_means
            if status == "independent":
                eff_n_for_se = float(n_blocks)
                se_method = "iid_blocks"
                warning = None

            elif status == "best_p":
                eff_n_for_se = float(max(1.0, n_blocks))
                se_method = "iid_blocks_best_p"
                warning = (
                    "Block means did not pass Ljung–Box. "
                    "Using best-p-value window and treating block means as independent."
                )
            elif status == "user_window":
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks"
                warning = (
                    "User-specified window used without independence verification. "
                    "SE computed using ESS on block means."
                )
            elif status == "too_few_blocks":
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks_too_few_blocks"
                warning = (
                    "Too few blocks to test Ljung–Box independence. "
                    "SE computed using ESS on block means."
                )
            else:
                # safety fallback (should never trigger)
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks_fallback"
                warning = (
                    "Unknown independence status. "
                    "Using ESS-corrected SE on block means."
                )
            se = (
                float(sd / np.sqrt(eff_n_for_se))
                if block_means.size >= 2
                else np.nan
            )
            ci = (
                (float(mu - 1.96 * se), float(mu + 1.96 * se))
                if np.isfinite(se)
                else (np.nan, np.nan)
            )

            # 5) Raw-series ESS value (existing)
            ess_val_raw = ess_res.get("results", {}).get(col, 10)
            if isinstance(ess_val_raw, dict):
                ess_val = ess_val_raw.get("effective_sample_size", None)
            else:
                ess_val = ess_val_raw

            stats[col] = {
                "mean": mu,
                "mean_uncertainty": se,  # SE from blocks
                "variance": var_val,  # Var(block_means)
                "confidence_interval": ci,
                "pm_std": (
                    (mu - se, mu + se) if np.isfinite(se) else (np.nan, np.nan)
                ),
                # bookkeeping
                "window_size": int(w),
                "n_short_averages": int(n_blocks),  # count of block means
                # IMPORTANT: standard key for Ensemble + diagnostics
                "ess_blocks": float(ess_blocks),
                "block_effective_n": float(ess_blocks),  # ESS on block means
                "se_effective_n": float(eff_n_for_se),  # ESS used for SE
                "se_method": se_method,
                # independence reporting
                "independence_status": status,
                "ljungbox_lags": lb.get("lags", []),
                "ljungbox_pvalues": lb.get("pvalues", []),
            }
            if warning:
                stats[col]["warning"] = warning

        # History + metadata
        op_options = dict(
            column_name=column_name,
            ddof=ddof,
            method=method,
            window_size=window_size,
        )
        self._add_history("compute_statistics", op_options)
        stats["metadata"] = _diagnostics_view(
            self._history, diagnostics=diagnostics
        )
        return to_native_types(stats)

    def cumulative_statistics(
        self,
        column_name=None,
        method="non-overlapping",
        window_size=None,
        diagnostics="compact",
    ):
        results = {}
        for col in self._get_columns(column_name):
            series = self.df[col].dropna()
            if series.empty:
                results[col] = {"error": f"No data for '{col}'"}
                continue
            est_win = self._estimate_window(col, series, window_size)
            proc = self._process_column(series, est_win, method)
            cm = proc.expanding().mean()
            cs = proc.expanding().std()
            cnt = proc.expanding().count()
            se = cs / np.sqrt(cnt)
            results[col] = {
                "cumulative_mean": cm.tolist(),
                "cumulative_uncertainty": cs.tolist(),
                "standard_error": se.tolist(),
                "window_size": int(est_win),
            }
        self._add_history(
            "cumulative_statistics",
            {
                "column_name": column_name,
                "method": method,
                "window_size": window_size,
            },
        )
        results["metadata"] = _diagnostics_view(
            self._history, diagnostics=diagnostics
        )
        return to_native_types(results)

    def additional_data(
        self,
        column_name=None,
        ddof=1,
        method="sliding",
        window_size=None,
        reduction_factor=0.1,
        diagnostics="compact",
    ):
        stats = self.cumulative_statistics(
            column_name,
            method=method,
            window_size=window_size,
            diagnostics="none",
        )
        results = {}
        cols = self._get_columns(column_name)
        for col in cols:
            if "cumulative_uncertainty" not in stats.get(col, {}):
                results[col] = {"error": f"No cumulative SEM for '{col}'"}
                continue
            series = self.df[col].dropna()
            est_win = self._estimate_window(col, series, window_size)
            cum_sem = np.array(stats[col]["cumulative_uncertainty"])
            n_cur = len(cum_sem)
            n_arr = np.arange(1, n_cur + 1)
            mask = np.isfinite(cum_sem)
            x = n_arr[mask]
            y = cum_sem[mask]
            if len(x) < 2:
                results[col] = {"error": "Not enough points for fit."}
                continue

            def model(n, A, p):
                return A / (n**p)

            A_est, p_est = curve_fit(model, x, y, p0=[1.0, 0.5])[0]
            p_est = abs(p_est)
            cur_sem = model(n_cur, A_est, p_est)
            tgt_sem = (1 - reduction_factor) * cur_sem
            n_tgt = (A_est / tgt_sem) ** (1 / p_est)
            add = n_tgt - n_cur
            if method == "non-overlapping":
                add *= est_win
            results[col] = {
                "A_est": float(A_est),
                "p_est": float(p_est),
                "n_current": int(n_cur),
                "current_sem": float(cur_sem),
                "target_sem": float(tgt_sem),
                "n_target": float(n_tgt),
                "additional_samples": int(math.ceil(add)),
                "window_size": int(est_win),
            }
        self._add_history(
            "additional_data",
            {
                "column_name": column_name,
                "ddof": ddof,
                "method": method,
                "window_size": window_size,
                "reduction_factor": reduction_factor,
            },
        )
        results["metadata"] = _diagnostics_view(
            self._history, diagnostics=diagnostics
        )
        return to_native_types(results)

    # ===================== Stationarity & Independence =====================

    def is_stationary(self, columns, diagnostics="compact"):
        self._add_history("is_stationary", {"columns": columns})
        cols = [columns] if isinstance(columns, str) else list(columns)
        out = {}
        for c in cols:
            try:
                p = adfuller(self.df[c].dropna(), autolag="AIC")[1]
                out[c] = p < 0.05
            except Exception as e:
                out[c] = f"Error: {e}"
        # Display view only if requested (callers often want just the dict)
        return out

    def make_stationary(self, col, n_pts_orig, workflow):
        """
        Attempt to make the data stream into being stationary by removing an initial
        fraction of data.

        Parameters
        ----------
        col : str
        n_pts_orig : int
        workflow : RobustWorkflow

        Returns
        -------
        self : DataStream
        stationary : bool
        """
        stationary = self.is_stationary([col])[
            col
        ]  # is_stationary() returns dictionary. The value for key qoi tells us if it is stationary
        n_pts = len(self.df)

        ds = self

        n_dropped = 0
        while (
            not stationary
            and not workflow._operate_safe
            and n_pts > workflow._n_pts_min
            and n_pts > workflow._n_pts_frac_min * n_pts_orig
        ):
            # See if we get a stationary stream if we drop some initial fraction of the data
            n_drop = int(n_pts * workflow._drop_fraction)
            df_shortened = ds.df.iloc[n_drop:]
            ds = DataStream(df_shortened)
            n_pts = len(ds.df)
            n_dropped = n_pts_orig - n_pts
            stationary = ds.is_stationary([col])[col]

            if workflow._verbosity > 0:
                if stationary:
                    print(
                        f"Data stream was not stationary, but is stationary after dropping first {n_dropped} points."
                    )
                else:
                    print(
                        f"Data stream is not stationary, even after dropping first {n_dropped} points."
                    )

        return ds, stationary

    # === Compatibility wrappers for legacy tests ===

    def mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Legacy wrapper for test compatibility. Returns only mean (not dict).
        """
        results = self._mean(
            column_name=column_name, method=method, window_size=window_size
        )
        if column_name is None:
            if "time" in self.df.columns:
                column_name = "time"
            else:
                raise ValueError(
                    "No 'time' column and no column_name provided."
                )
        times = self.df[column_name].values
        if len(times) < 2:
            raise ValueError(f"Not enough values in column '{column_name}'.")

        steps = np.diff(times)
        rounded = np.round(steps / tol) * tol
        uniq = np.unique(rounded)

        # contiguous blocks (for internal attachment only)
        step_blocks = []
        prev = rounded[0]
        start = 0
        for i, s in enumerate(rounded[1:], 1):
            if not np.isclose(s, prev, atol=tol):
                step_blocks.append(
                    {
                        "step": float(prev),
                        "start": int(start),
                        "end": int(i - 1),
                        "block_len": int(i - start),
                    }
                )
                start = i
                prev = s
        step_blocks.append(
            {
                "step": float(prev),
                "start": int(start),
                "end": int(len(rounded) - 1),
                "block_len": int(len(rounded) - start),
            }
        )

        status = "NotUniform"
        if len(uniq) == 1:
            status = "AllEqual"
        elif (
            len(uniq) == 2
            and np.allclose(rounded[:-1], rounded[0], atol=tol)
            and not np.isclose(rounded[-1], rounded[0], atol=tol)
        ):
            status = "AllEqualButLast"

        # Minimal options stored in history (compact by design)
        self._add_history(
            "check_time_steps_uniformity",
            {
                "column_name": column_name,
                "tol": tol,
                "status": status,
                "interp_kind": (
                    interp_kind if status == "AllEqualButLast" else None
                ),
            },
        )

        if status == "AllEqual":
            ds_copy = self.__class__(
                self.df.copy(), _history=self._history.copy()
            )
            ds_copy._uniformity_result = {
                "status": status,
                "unique_steps": uniq.tolist(),
                "num_unique": int(len(uniq)),
                "step_blocks": step_blocks,
                "total_steps": int(len(steps)),
            }
            if print_details:
                print(
                    f"[{column_name}] Uniform: step={uniq}, Nsteps={len(steps)}"
                )
            return ds_copy

        if status == "AllEqualButLast":
            regular_step = rounded[0]
            n_pts = int(np.round((times[-1] - times[0]) / regular_step)) + 1
            regular_times = times[0] + regular_step * np.arange(n_pts)

            interp_df = {column_name: regular_times}
            for col in self.df.columns:
                if col == column_name:
                    continue
                kind = interp_kind if len(times) >= 4 else "linear"
                f = interp1d(
                    times,
                    self.df[col].values,
                    kind=kind,
                    fill_value="extrapolate",
                )
                interp_df[col] = f(regular_times)

            ds_new = self.__class__(
                pd.DataFrame(interp_df), _history=self._history.copy()
            )
            ds_new._uniformity_result = {
                "status": status,
                "unique_steps": uniq.tolist(),
                "num_unique": int(len(uniq)),
                "step_blocks": step_blocks,
                "total_steps": int(len(steps)),
            }
            if print_details:
                print(
                    f"[{column_name}] AllEqualButLast -> interpolated to regular grid (step={regular_step}, N={n_pts})."
                )
            return ds_new

        # NotUniform -> return empty DataStream (but with full internal history)
        empty = self.df.iloc[0:0].copy()
        ds_empty = self.__class__(empty, _history=self._history.copy())
        ds_empty._uniformity_result = {
            "status": status,
            "unique_steps": uniq.tolist(),
            "num_unique": int(len(uniq)),
            "step_blocks": step_blocks,
            "total_steps": int(len(steps)),
        }
        if print_details:
            print(
                f"[{column_name}] NotUniform: irregular steps; returning empty stream."
            )
        return ds_empty

    def test_block_independence(
        self,
        column_name=None,
        method="non-overlapping",
        window_size=None,
        max_lag=10,
        alpha=0.05,
        verbose=True,
        start_time=None,
        use_auto_lag=True,
        diagnostics="compact",
    ):
        """
        Ljung-Box + ACF on block means for independence.
        Returns dict for a single column; if column_name is None/list, returns dict of dicts.
        """
        # Multi-column dispatch
        if column_name is None or (
            isinstance(column_name, list) and len(column_name) > 1
        ):
            cols = (
                [c for c in self.df.columns if c != "time"]
                if column_name is None
                else [
                    c
                    for c in column_name
                    if c in self.df.columns and c != "time"
                ]
            )
            out = {}
            for c in cols:
                try:
                    out[c] = self.test_block_independence(
                        column_name=c,
                        method=method,
                        window_size=window_size,
                        max_lag=max_lag,
                        alpha=alpha,
                        verbose=verbose,
                        start_time=start_time,
                        use_auto_lag=use_auto_lag,
                        diagnostics="none",
                    )
                except Exception as e:
                    out[c] = {"error": str(e)}
            return out

        if isinstance(column_name, list):
            column_name = column_name[0]

        df = (
            self.df
            if start_time is None
            else self.df[self.df["time"] >= start_time]
        )
        data = df[column_name].dropna()
        if data.empty:
            raise ValueError(
                "No data available for this column in the specified range."
            )

        est_win = (
            self._estimate_window(column_name, data, window_size)
            if window_size is None
            else window_size
        )
        proc = self._process_column(data, est_win, method)
        block_means = proc.values
        block_idx = proc.index.values
        n_blocks = len(block_means)

        # Choose lags
        if use_auto_lag:
            auto_lag = min(20, max(1, n_blocks // 4))
        else:
            auto_lag = max_lag

        if n_blocks < 2:
            acf_vals = np.array([1.0])
            confint = np.nan
            pvalue = np.nan
            indep = None
            pvalue_auto = None
            indep_auto = None
            msg = "Not enough blocks (n_blocks < 2) for independence test."
            if verbose:
                print(msg)
        else:
            chosen_lag = min(n_blocks - 1, max_lag)
            chosen_auto = min(n_blocks - 1, auto_lag)
            acf_vals = acf(block_means, nlags=chosen_lag, fft=False)
            confint = 1.96 / np.sqrt(n_blocks)
            lb = acorr_ljungbox(block_means, lags=[chosen_lag], return_df=True)
            pvalue = float(lb["lb_pvalue"].iloc[0])
            indep = bool(pvalue > alpha)

            if use_auto_lag and chosen_auto != chosen_lag:
                lb2 = acorr_ljungbox(
                    block_means, lags=[chosen_auto], return_df=True
                )
                pvalue_auto = float(lb2["lb_pvalue"].iloc[0])
                indep_auto = bool(pvalue_auto > alpha)
            else:
                pvalue_auto = None
                indep_auto = None

            if n_blocks <= 5:
                msg = "Not enough blocks (n_blocks <= 5). Use results with extreme caution."
                if verbose:
                    print(msg)
            else:
                msg = None
                if verbose:
                    print(
                        f"Block means (n={n_blocks}, win={est_win}) ACF lags 1..{chosen_lag}; Ljung-Box p={pvalue:.4f}"
                    )

        # Log compact options in history
        self._add_history(
            "test_block_independence",
            {
                "column_name": column_name,
                "method": method,
                "window_size": window_size,
                "est_win": est_win,
                "max_lag": max_lag,
                "auto_lag": auto_lag,
                "alpha": alpha,
                "start_time": start_time,
                "ljungbox_pvalue": (
                    float(pvalue) if not np.isnan(pvalue) else None
                ),
                "ljungbox_pvalue_auto": pvalue_auto,
                "independent": indep if indep is not None else None,
                "independent_auto": indep_auto,
            },
        )

        return {
            "acf": acf_vals,
            "acf_confint": confint,
            "ljungbox_pvalue": pvalue,
            "ljungbox_pvalue_auto": pvalue_auto,
            "independent": indep,
            "independent_auto": indep_auto,
            "block_means": block_means,
            "block_indices": block_idx,
            "metadata": _diagnostics_view(
                self._history, diagnostics=diagnostics
            ),
            "message": msg,
        }
