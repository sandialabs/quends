"""
Legacy DataStream helpers kept for reference.

These are intentionally private and not part of the public API.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts
import statsmodels.tsa.stattools as ststls
from scipy.stats import norm
from statsmodels.robust.scale import mad
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from .data_stream import DataStream


def _trim_sss_start_old(ds: DataStream, col, workflow):
    """
    Legacy SSS trimming (pre-vectorized).
    """
    n_pts = len(ds.df)
    max_lag = int(workflow._max_lag_frac * n_pts)

    acf_vals = ststls.acf(ds.df[col].dropna().values, nlags=max_lag)

    if workflow._verbosity > 1:
        plt.figure(figsize=(10, 6))
        plt.stem(range(len(acf_vals)), acf_vals)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Function")
        plt.grid()
        plt.show()
        plt.close()

    z_critical = sts.norm.ppf(1 - workflow._autocorr_sig_level / 2)
    conf_interval = z_critical / np.sqrt(n_pts)
    significant_lags = np.where(np.abs(acf_vals[1:]) > conf_interval)[0]
    acf_sum = np.sum(np.abs(acf_vals[1:][significant_lags]))
    decor_length = int(np.ceil(1 + 2 * acf_sum))

    decor_index = min(int(workflow._decor_multiplier * decor_length), max_lag)

    if workflow._verbosity > 0:
        print(
            f"stats decorrelation length {decor_length} gives smoothing window of {decor_index} points."
        )

    rolling_window = max(3, decor_index)
    col_smoothed = ds.df[col].rolling(window=rolling_window).mean()
    col_sm_flld = col_smoothed.bfill()
    df_smoothed = pd.DataFrame({"time": ds.df["time"], col: col_sm_flld})

    std_dev_till_end = np.empty((n_pts,), dtype=float)
    for i in range(n_pts):
        std_dev_till_end[i] = np.std(ds.df[col].iloc[i:])
    std_dev_till_end_series = pd.Series(std_dev_till_end, index=ds.df.index)
    std_dev_smoothed = std_dev_till_end_series.rolling(
        window=workflow._final_smoothing_window
    ).mean()
    std_dev_sm_flld = std_dev_smoothed.bfill()

    df_std_dev = pd.DataFrame(
        {"time": ds.df["time"], col + "_std_till_end": std_dev_sm_flld}
    )

    smoothed_start_time = df_smoothed["time"].iloc[rolling_window - 1]

    if workflow._verbosity > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(
            ds.df["time"],
            ds.df[col],
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

    n_pts_smoothed = len(df_smoothed)
    mean_vals = np.empty((n_pts_smoothed,), dtype=float)
    for i in range(n_pts_smoothed):
        mean_vals[i] = np.mean(df_smoothed[col].iloc[i:])

    deviation_arr = np.abs(df_smoothed[col] - mean_vals)

    deviation_series = pd.Series(deviation_arr, index=ds.df.index)
    deviation_smoothed = deviation_series.rolling(
        window=workflow._final_smoothing_window
    ).mean()
    deviation_sm_flld = deviation_smoothed.bfill()
    deviation = pd.DataFrame(
        {"time": ds.df["time"], col + "_deviation": deviation_sm_flld}
    )

    tol_fac = workflow._std_dev_frac * (
        df_std_dev[col + "_std_till_end"]
        + workflow._fudge_fac * abs(mean_vals[0])
    )
    tolerance = tol_fac * np.abs(mean_vals)

    within_tolerance_all = deviation[col + "_deviation"] <= tolerance
    within_tolerance = within_tolerance_all & (
        df_smoothed["time"] >= smoothed_start_time
    )
    sss_index = np.where(within_tolerance)[0]

    crit_met_index = None
    if len(sss_index) > 0:
        for idx in sss_index:
            if np.all(within_tolerance[idx:]):
                crit_met_index = idx
                break

    if crit_met_index is not None:
        criterion_time = df_smoothed["time"].iloc[crit_met_index]
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
                x=sss_start_time,
                color="r",
                linestyle="--",
                label="Start SSS",
            )
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("Deviation and Tolerance vs. Time")
            plt.legend()
            plt.grid()
            plt.show()
            plt.close()

        trimmed_df = ds.df[ds.df["time"] >= sss_start_time]
        trimmed_df = trimmed_df.reset_index(drop=True)
        trimmed_stream = DataStream(trimmed_df)

    else:
        if workflow._verbosity > 0:
            print("No SSS found based on behavior of mean of smoothed signal.")
        trimmed_stream = pd.DataFrame(columns=["time", "flux"])

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


def _find_steady_state_std_old(data, column_name, window_size=10, robust=True):
    time_vals = data["time"].values
    x = data[column_name].values
    for i in range(len(x) - window_size + 1):
        rem = x[i:]
        if robust:
            center = np.median(rem)
            scale = mad(rem)
            if scale == 0:
                scale = np.std(rem)
            if scale == 0:
                continue
        else:
            center = np.mean(rem)
            scale = np.std(rem)
        within_1 = np.mean(np.abs(rem - center) <= 1 * scale)
        within_2 = np.mean(np.abs(rem - center) <= 2 * scale)
        within_3 = np.mean(np.abs(rem - center) <= 3 * scale)
        if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
            return time_vals[i]
    return None


def _estimate_window_old(ds: DataStream, col, column_data, window_size):
    if window_size is None:
        ess_info = ds.effective_sample_size(
            column_names=col, method="geyer", diagnostics="none"
        )
        ess_val = 10
        try:
            entry = ess_info.get("results", {}).get(col, {})
            if isinstance(entry, dict):
                ess_val = int(max(1, entry.get("effective_sample_size", 10)))
            else:
                ess_val = int(max(1, entry))
        except Exception:
            pass
        return max(5, len(column_data) // ess_val)
    return int(window_size)


def _process_column_old(column_data, estimated_window, method):
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
        raise ValueError("Invalid method. Choose 'sliding' or 'non-overlapping'.")


def _mean_uncertainty_old(
    ds: DataStream,
    column_name=None,
    ddof=1,
    method="non-overlapping",
    window_size=None,
):
    results = {}
    for col in ds._get_columns(column_name):
        data = ds.df[col].dropna()
        if data.empty:
            results[col] = {
                "mean_uncertainty": np.nan,
                "window_size": np.nan,
                "error": f"No data for '{col}'",
            }
            continue

        est_win = ds._estimate_window(col, data, window_size)
        proc = ds._process_column(data, est_win, method)
        block_means = proc.values

        if len(block_means) >= 3:
            r = acf(block_means, nlags=max(1, len(block_means) // 4), fft=False)
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


def _compute_statistics_old(
    ds: DataStream,
    column_name=None,
    ddof=1,
    method="non-overlapping",
    window_size=None,
    diagnostics="compact",
):
    stats = {}
    cols = ds._get_columns(column_name)

    mean_res = ds._mean(column_name, method=method, window_size=window_size)
    mu_res = ds._mean_uncertainty(
        column_name, ddof=ddof, method=method, window_size=window_size
    )
    ci_res = ds._confidence_interval(
        column_name, ddof=ddof, method=method, window_size=window_size
    )
    ess_res = ds.effective_sample_size(
        column_names=column_name, method="geyer", diagnostics="none"
    )
    var_res = ds._variance(
        column_name, ddof=ddof, method=method, window_size=window_size
    )
    short_res = ds._short_term_counts(
        column_name, method=method, window_size=window_size
    )

    for col in cols:
        try:
            ind = ds.test_block_independence(
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
            warn = (
                "Short-term averages failed independence (autocorrelation detected). Use stats with caution."
            )
        elif independent is None:
            warn = (
                "Block independence undetermined (insufficient data). Use stats with caution."
            )

        data = ds.df[col].dropna()
        if data.empty:
            stats[col] = {"error": f"No data for '{col}'"}
            continue

        mu = mean_res[col]["mean"]
        se = mu_res[col]["mean_uncertainty"]
        ci = ci_res[col]["confidence_interval"]
        est_win = mean_res[col]["window_size"]
        entry = ess_res.get("results", {}).get(col, {})
        if isinstance(entry, dict):
            ess_val = entry.get("effective_sample_size", 10)
        else:
            ess_val = entry

        var_val = var_res.get(col, {}).get("variance", np.nan)
        block_ess = var_res.get(col, {}).get("effective_n_blocks", np.nan)
        n_short = short_res.get(col, {}).get("n_short_averages", np.nan)

        stats[col] = {
            "mean": mu,
            "mean_uncertainty": se,
            "variance": var_val,
            "confidence_interval": ci,
            "pm_std": (mu - se, mu + se),
            "effective_sample_size": ess_val,
            "block_effective_n": block_ess,
            "n_short_averages": n_short,
            "window_size": est_win,
        }
        if warn:
            stats[col]["warning"] = warn

    ds._add_history(
        "compute_statistics",
        {
            "column_name": column_name,
            "ddof": ddof,
            "method": method,
            "window_size": window_size,
        },
    )
    stats["metadata"] = ds.get_metadata(diagnostics=diagnostics)
    return stats


def _make_stationary_old(ds: DataStream, col, n_pts_orig, workflow):
    stationary = ds.is_stationary([col]).get("results", {}).get(col, False)
    n_pts = len(ds.df)

    n_dropped = 0
    while (
        not stationary
        and not workflow._operate_safe
        and n_pts > workflow._n_pts_min
        and n_pts > workflow._n_pts_frac_min * n_pts_orig
    ):
        n_drop = int(n_pts * workflow._drop_fraction)
        df_shortened = ds.df.iloc[n_drop:]
        ds = DataStream(df_shortened)
        n_pts = len(ds.df)
        n_dropped = n_pts_orig - n_pts
        stationary = ds.is_stationary([col]).get("results", {}).get(col, False)

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
