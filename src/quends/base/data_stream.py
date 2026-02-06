import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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

    def get_metadata(self, diagnostics="compact"):
        return _diagnostics_view(self._history, diagnostics=diagnostics)

    # ---- basic helpers ----
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

    def _time_values_for_series(
        self, series: pd.Series
    ) -> Optional[np.ndarray]:
        if "time" not in self.df.columns:
            return None
        try:
            return self.df.loc[series.index, "time"].to_numpy(dtype=float)
        except Exception:
            return None

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
        is_stat = bool(
            stationary_result.get("results", {}).get(column_name, False)
            if isinstance(stationary_result, dict)
            else stationary_result
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
                f"Column '{column_name}' is not stationary. Steady-state trimming requires stationary data."
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
                "Invalid method. Choose 'std', 'threshold', 'rolling_variance', "
                "'self_consistent', or 'iqr'."
            )
            new_history.append({"operation": "trim", "options": options})
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

        options["sss_start"] = sss
        new_history.append({"operation": "trim", "options": options})

        if sss is None:
            return DataStream(self.df.iloc[0:0].copy(), _history=new_history)

        trimmed_df = self.df.loc[
            self.df["time"] >= sss, ["time", column_name]
        ].reset_index(drop=True)
        return DataStream(trimmed_df, _history=new_history)

    def trim_sss_start(self, col, workflow):
        """
        Identify and trim the signal to the start of the Statistical Steady State (SSS).

        Returns
        -------
        DataStream
            A new DataStream trimmed to SSS start, or an empty DataStream if none found.
        """
        # ---- basic safety ----
        if "time" not in self.df.columns or col not in self.df.columns:
            empty_df = self.df.iloc[0:0].copy()
            return DataStream(empty_df, _history=self._history.copy())

        n_pts = len(self.df)
        if n_pts < 3:
            empty_df = self.df.iloc[0:0].copy()
            return DataStream(empty_df, _history=self._history.copy())

        # ---- decorrelation length / smoothing window ----
        max_lag = int(workflow._max_lag_frac * n_pts)
        max_lag = max(1, min(max_lag, n_pts - 1))

        # Use imported `acf` directly (you already imported: from statsmodels.tsa.stattools import acf)
        acf_vals = acf(self.df[col].dropna().values, nlags=max_lag, fft=False)

        if workflow._verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.stem(range(len(acf_vals)), acf_vals)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Autocorrelation Function")
            plt.grid()
            plt.show()
            plt.close()

        # Use imported `norm` directly (you already imported: from scipy.stats import norm)
        z_critical = norm.ppf(1 - workflow._autocorr_sig_level / 2)
        conf_interval = z_critical / np.sqrt(n_pts)

        significant_lags = np.where(np.abs(acf_vals[1:]) > conf_interval)[0]
        acf_sum = (
            np.sum(np.abs(acf_vals[1:][significant_lags]))
            if significant_lags.size
            else 0.0
        )
        decor_length = int(np.ceil(1 + 2 * acf_sum))

        decor_index = min(
            int(workflow._decor_multiplier * decor_length), max_lag
        )
        rolling_window = max(3, decor_index)

        if workflow._verbosity > 0:
            print(
                f"stats decorrelation length {decor_length} gives smoothing window of {decor_index} points."
            )

        # ---- smooth signal ----
        col_smoothed = self.df[col].rolling(window=rolling_window).mean()
        col_sm_flld = col_smoothed.bfill()

        df_smoothed = pd.DataFrame({"time": self.df["time"], col: col_sm_flld})
        smoothed_start_time = df_smoothed["time"].iloc[rolling_window - 1]

        # ---- std-dev till end (smoothed) ----
        raw = self.df[col].to_numpy(dtype=float)
        rev = raw[::-1]
        csum = np.cumsum(rev)[::-1]
        csum2 = np.cumsum(rev**2)[::-1]
        counts = np.arange(n_pts, 0, -1, dtype=float)
        mean_tail = csum / counts
        var_tail = (csum2 / counts) - (mean_tail**2)
        var_tail = np.maximum(var_tail, 0.0)
        std_dev_till_end = np.sqrt(var_tail)

        std_dev_till_end_series = pd.Series(
            std_dev_till_end, index=self.df.index
        )

        std_dev_smoothed = std_dev_till_end_series.rolling(
            window=workflow._final_smoothing_window
        ).mean()
        std_dev_sm_flld = std_dev_smoothed.bfill()

        df_std_dev = pd.DataFrame(
            {"time": self.df["time"], f"{col}_std_till_end": std_dev_sm_flld}
        )

        if workflow._verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.df["time"],
                self.df[col],
                label="Original Signal",
                alpha=0.5,
            )
            plt.plot(
                df_smoothed["time"], df_smoothed[col], label="Smoothed Signal"
            )
            plt.plot(
                df_std_dev["time"],
                df_std_dev[f"{col}_std_till_end"],
                label="Smoothed Std Dev Till End",
            )
            plt.axvline(
                x=smoothed_start_time,
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

        # ---- mean of remaining smoothed signal ----
        x_sm = df_smoothed[col].to_numpy(dtype=float)
        t_sm = df_smoothed["time"].to_numpy(dtype=float)

        n_pts_smoothed = len(df_smoothed)
        rev_sm = x_sm[::-1]
        csum_sm = np.cumsum(rev_sm)[::-1]
        counts_sm = np.arange(n_pts_smoothed, 0, -1, dtype=float)
        mean_vals = csum_sm / counts_sm

        deviation_arr = np.abs(x_sm - mean_vals)

        # IMPORTANT: index must match df_smoothed, not self.df
        deviation_series = pd.Series(deviation_arr, index=df_smoothed.index)

        deviation_smoothed = deviation_series.rolling(
            window=workflow._final_smoothing_window
        ).mean()
        deviation_sm_flld = deviation_smoothed.bfill()

        deviation = pd.DataFrame(
            {"time": df_smoothed["time"], f"{col}_deviation": deviation_sm_flld}
        )

        # ---- tolerance rule ----
        # tol_fac is already an absolute tolerance scale; DO NOT multiply by |mean_vals| again.
        tol_fac = workflow._std_dev_frac * (
            df_std_dev[f"{col}_std_till_end"].to_numpy(dtype=float)
            + workflow._fudge_fac * abs(mean_vals[0])
        )
        tolerance = tol_fac  # same length as time

        within_tolerance_all = (
            deviation[f"{col}_deviation"].to_numpy(dtype=float) <= tolerance
        )
        within_tolerance = within_tolerance_all & (t_sm >= smoothed_start_time)

        sss_index = np.where(within_tolerance)[0]

        crit_met_index = None
        if len(sss_index) > 0:
            for idx in sss_index:
                if np.all(within_tolerance[idx:]):
                    crit_met_index = int(idx)
                    break
        # ---- if SSS found ----
        if crit_met_index is not None:
            criterion_time = float(t_sm[crit_met_index])

            true_sss_start_index = max(
                0,
                int(
                    crit_met_index
                    - workflow._smoothing_window_correction * rolling_window
                ),
            )
            sss_start_time = float(t_sm[true_sss_start_index])

            if workflow._verbosity > 0:
                print(f"Index where criterion is met: {crit_met_index}")
                print(f"Rolling window: {rolling_window}")
                print(f"time where criterion is met: {criterion_time}")
                print(
                    f"time at start of SSS (adjusted for rolling window): {sss_start_time}"
                )

            if workflow._verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(t_sm, deviation[f"{col}_deviation"], label="Deviation")
                plt.plot(t_sm, tolerance, label="Tolerance")
                plt.axvline(
                    x=criterion_time, linestyle="--", label="Criterion Met"
                )
                plt.axvline(x=sss_start_time, linestyle="--", label="Start SSS")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title("Deviation and Tolerance vs. Time")
                plt.legend()
                plt.grid()
                plt.show()
                plt.close()

            trimmed_df = self.df[self.df["time"] >= sss_start_time].reset_index(
                drop=True
            )

            # preserve history + log
            new_history = self._history.copy()
            new_history.append(
                {
                    "operation": "trim_sss_start",
                    "options": {
                        "col": col,
                        "sss_start_time": sss_start_time,
                        "rolling_window": int(rolling_window),
                        "decor_length": int(decor_length),
                        "decor_index": int(decor_index),
                    },
                }
            )
            return DataStream(trimmed_df, _history=new_history)

        # ---- if no SSS found: return EMPTY DataStream (not DataFrame) ----
        if workflow._verbosity > 0:
            print("No SSS found based on behavior of mean of smoothed signal.")

        if workflow._verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(t_sm, deviation[f"{col}_deviation"], label="Deviation")
            plt.plot(t_sm, tolerance, label="Tolerance")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("Deviation and Tolerance vs. Time")
            plt.legend()
            plt.grid()
            plt.show()
            plt.close()

        empty_df = self.df.iloc[0:0].copy()
        new_history = self._history.copy()
        new_history.append(
            {
                "operation": "trim_sss_start",
                "options": {"col": col, "sss_start_time": None},
            }
        )
        return DataStream(empty_df, _history=new_history)

    def make_stationary(self, col, n_pts_orig, workflow):
        """
        Attempt to make the data stream stationary by repeatedly dropping an initial fraction.

        Returns
        -------
        (DataStream, bool)
            (new_stream, stationary_flag)
        """
        stationary = self.is_stationary([col]).get("results", {}).get(col, False)
        n_pts = len(self.df)

        ds = DataStream(self.df.copy(), _history=self._history.copy())
        n_dropped = 0

        while (
            (not stationary)
            and (not workflow._operate_safe)  # keep your original semantics
            and (n_pts > workflow._n_pts_min)
            and (n_pts > workflow._n_pts_frac_min * n_pts_orig)
        ):
            n_drop = int(n_pts * workflow._drop_fraction)
            n_drop = max(1, n_drop)  # ensure progress

            df_shortened = ds.df.iloc[n_drop:].reset_index(drop=True)

            ds = DataStream(df_shortened, _history=ds._history.copy())
            n_pts = len(ds.df)
            n_dropped = n_pts_orig - n_pts
            stationary = (
                ds.is_stationary([col]).get("results", {}).get(col, False)
            )

            ds._history.append(
                {
                    "operation": "make_stationary_drop",
                    "options": {
                        "col": col,
                        "n_drop": int(n_drop),
                        "n_dropped_total": int(n_dropped),
                        "n_pts_now": int(n_pts),
                        "stationary": bool(stationary),
                    },
                }
            )

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
                out[col] = {
                    "effective_sample_size": None,
                    "message": f"Column '{col}' not found.",
                }
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
            out[col] = {
                "effective_sample_size": int(math.ceil(ess)),
                "message": None,
            }
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
        time_values = self._time_values_for_series(series)
        proc = self._process_column(
            series, est_win, method, time_values=time_values
        )
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
          3) Ljung–Box test on block means; if fail -> increase w via +1 (conservative)
          4) enforce minimum block count B_min when possible
          5) fallback to best p-value if never passes

        Returns
        -------
        (chosen_w:int, info:dict)
        """
        x = series.dropna()
        time_values = self._time_values_for_series(x)
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
        w0 = int(max(w_min, math.ceil(c0 * tau)))
        w = int(w0)

        # Ensure we *try* to keep at least B_min blocks if possible:
        # i.e., w <= n//B_min. If n//B_min < w_min, we can't enforce.
        w_cap_for_blocks = (n // B_min) if (B_min and n // B_min >= 1) else n
        cap_applied = False
        if w_cap_for_blocks >= w_min and w > w_cap_for_blocks:
            w = min(w, w_cap_for_blocks)
            cap_applied = True

        best = {
            "w": w,
            "passed": False,
            "p_min": -np.inf,
            "details": None,
            "lb": {"lags": [], "pvalues": []},
        }

        for it in range(int(max_iter)):
            proc = self._process_column(
                x,
                estimated_window=w,
                method="non-overlapping",
                time_values=time_values,
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
                note = None
                if cap_applied:
                    note = (
                        "Initial window capped to preserve >=B_min blocks; cap lifted after first failure."
                    )
                return int(w), {
                    "status": "independent",
                    "tau_int": float(tau),
                    "chosen_w": int(w),
                    "passed": True,
                    "iter": int(it),
                    "lb": det,
                    "note": note,
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
            note = (
                "Did not reach Ljung–Box pass; using best observed p-value window."
            )
            if cap_applied:
                note += (
                    " Initial window capped to preserve >=B_min blocks; cap lifted after first failure."
                )
            return int(best["w"]), {
                "status": "best_p",
                "tau_int": float(tau),
                "chosen_w": int(best["w"]),
                "passed": False,
                "iter": int(max_iter),
                "lb": best["lb"],
                "note": note,
            }
        # Otherwise: too few blocks to test
        note = "Window growth left <2 blocks; cannot test independence."
        if cap_applied:
            note += (
                " Initial window capped to preserve >=B_min blocks; cap lifted after first failure."
            )
        return int(w), {
            "status": "too_few_blocks",
            "tau_int": float(tau),
            "chosen_w": int(w),
            "passed": False,
            "lb": {"lags": [], "pvalues": []},
            "note": note,
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

    def _process_column(
        self,
        column_data,
        estimated_window,
        method,
        time_values: Optional[np.ndarray] = None,
    ):
        # If time_values are provided, block indices are time-based (block center time).
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

            # drop remainder
            x2 = x[: n_blocks * w].reshape(n_blocks, w)
            block_means = x2.mean(axis=1)

            if time_values is not None:
                t = time_values[: n_blocks * w].reshape(n_blocks, w)
                idx = t.mean(axis=1)
            else:
                idx = (np.arange(n_blocks) * w) + (w // 2)
            return pd.Series(block_means, index=idx)

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
            time_values = self._time_values_for_series(data)
            proc = self._process_column(
                data, est_win, method, time_values=time_values
            )
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
            time_values = self._time_values_for_series(series)
            proc = self._process_column(
                series, est_win, method, time_values=time_values
            )
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
            time_values = self._time_values_for_series(series)
            proc = self._process_column(
                series, est_win, method, time_values=time_values
            )
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

            time_values = self._time_values_for_series(series)
            proc = self._process_column(
                series, est_win, method, time_values=time_values
            )
            n_blocks = int(len(proc))

            results[col] = {
                "n_short_averages": n_blocks,
                "window_size": int(est_win),
                "n_blocks": n_blocks,
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
            else:
                w = int(window_size)
                status = "user_window"
                lb = {"lags": [], "pvalues": []}

            # 2) Block/sliding means ONCE
            time_values = self._time_values_for_series(series)
            proc = self._process_column(
                series,
                estimated_window=w,
                method=method,
                time_values=time_values,
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
            ess_entry = ess_res.get("results", {}).get(col, {})
            ess_val = (
                ess_entry.get("effective_sample_size", None)
                if isinstance(ess_entry, dict)
                else ess_entry
            )

            stats[col] = {
                "mean": mu,
                "mean_uncertainty": se,  # SE from blocks
                "variance": var_val,  # Var(block_means)
                "confidence_interval": ci,
                "pm_std": (
                    (mu - se, mu + se) if np.isfinite(se) else (np.nan, np.nan)
                ),
                "effective_sample_size": (
                    int(ess_val) if ess_val is not None else None
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
        metadata = _diagnostics_view(self._history, diagnostics=diagnostics)
        return {
            "results": to_native_types(stats),
            "metadata": metadata,
        }

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
            time_values = self._time_values_for_series(series)
            proc = self._process_column(
                series, est_win, method, time_values=time_values
            )
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
        results = {}
        errors = {}
        for c in cols:
            try:
                p = adfuller(self.df[c].dropna(), autolag="AIC")[1]
                results[c] = bool(p < 0.05)
            except Exception as e:
                results[c] = False
                errors[c] = str(e)
        metadata = {"errors": errors} if errors else {}
        return {
            "results": results,
            "metadata": metadata if diagnostics != "none" else {},
        }

    def check_time_steps_uniformity(
        self,
        column_name=None,
        tol=1e-8,
        print_details=True,
        interp_kind="cubic",
        diagnostics="compact",
    ):
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

        # NotUniform -> return a copy and let caller decide how to proceed
        ds_copy = self.__class__(self.df.copy(), _history=self._history.copy())
        ds_copy._uniformity_result = {
            "status": status,
            "unique_steps": uniq.tolist(),
            "num_unique": int(len(uniq)),
            "step_blocks": step_blocks,
            "total_steps": int(len(steps)),
        }
        if print_details:
            print(
                f"[{column_name}] NotUniform: irregular steps; returning original stream."
            )
        return ds_copy

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
        time_values = self._time_values_for_series(data)
        proc = self._process_column(
            data, est_win, method, time_values=time_values
        )
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
