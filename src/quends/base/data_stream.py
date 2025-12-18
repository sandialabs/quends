import math

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
        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
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
        else:
            options["message"] = (
                "Invalid method. Choose 'std', 'threshold', or 'rolling_variance'."
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

    @staticmethod
    def find_steady_state_std(data, column_name, window_size=10, robust=True):
        time_vals = data["time"].values
        x = data[column_name].values
        for i in range(len(x) - window_size + 1):
            rem = x[i:]
            if robust:
                center = np.median(rem)
                scale = mad(rem)
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
            out[col] = int(math.ceil(ess))

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

    # ===================== Block/Window Stats =====================

    def _estimate_window(self, col, column_data, window_size):
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

            #    Recompute block means (cheap) to get their std with your ddof
            proc = self._process_column(series, est_win, method)
            block_means = proc.values

            unc = float(np.std(block_means, ddof=ddof) / np.sqrt(eff_n))
            results[col] = {
                "mean_uncertainty": unc,
                "window_size": int(est_win),
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

    def compute_statistics(
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

            stats[col] = {
                "mean": mu,
                "mean_uncertainty": se,
                "confidence_interval": ci,
                "pm_std": (mu - se, mu + se),
                "effective_sample_size": ess_val,
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
