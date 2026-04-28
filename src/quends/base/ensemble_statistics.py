"""
ensemble_statistics.py
----------------------
Module-level statistical functions for ensemble analysis.

These functions implement the three ensemble-statistics techniques and operate
on plain lists of :class:`~quends.base.data_stream.DataStream` objects so that
workflow classes and the :class:`~quends.base.ensemble.Ensemble` class share
the same computation path without code duplication.

Technique 0 — Average-ensemble:
    Build one averaged trace and run DataStream statistics on it.

Technique 1 — Pooled-block (preferred for trimmed ensembles):
    Autotune window size per member until block means pass Ljung-Box
    independence, pool blocks across members, compute statistics on pooled
    series.

Technique 2 — Member-wise then aggregate:
    Call DataStream.compute_statistics() on each member; combine using
    inverse-variance weighting (fallback: simple mean).

Public API
----------
autotune_member_blocks_until_independent
pool_block_means
tech0_stats_for_col
tech1_pooled_stats_for_col
tech2_stats_for_col
compute_ensemble_statistics
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from quends.base.data_stream import DataStream
from quends.base.utils import _geyer_ess_on_blocks


# ---------------------------------------------------------------------------
# Technique 1 — helpers
# ---------------------------------------------------------------------------

def autotune_member_blocks_until_independent(
    ds: DataStream,
    col: str,
    window_size: Optional[int],
    method: str = "non-overlapping",
    lb_alpha: float = 0.05,
    lb_lags: Optional[int] = None,
    max_tries: int = 25,
    min_blocks: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    For one member, choose the smallest window so block means pass the
    Ljung-Box independence test, or use the window giving the best p-value.

    Parameters
    ----------
    ds : DataStream
    col : str
    window_size : int or None
        Starting window hint; auto-estimated if ``None``.
    method : {"non-overlapping", "sliding"}
    lb_alpha : float
        Ljung-Box significance level.
    lb_lags : int or None
        Lags for the test; auto-selected when ``None``.
    max_tries : int
        Maximum window increments to try.
    min_blocks : int
        Minimum blocks required to run Ljung-Box.

    Returns
    -------
    (block_means_array, meta_dict)
    """
    out_meta: Dict[str, Any] = {
        "status": None,
        "window_used": None,
        "n_blocks": 0,
        "ljungbox_pvalue": np.nan,
        "ljungbox_lags": 0,
        "tries": 0,
        "best_pvalue": -np.inf,
        "best_window": None,
        "warnings": [],
    }

    if col not in ds.data.columns:
        out_meta["status"] = "best_p"
        out_meta["warnings"].append("column_missing")
        return np.array([], dtype=float), out_meta

    series = ds.data[col].dropna()
    n = int(series.size)
    if n < 4:
        out_meta["status"] = "best_p"
        out_meta["warnings"].append("too_few_samples")
        return np.array([], dtype=float), out_meta

    w = int(max(2, ds._estimate_window(col, series, window_size)))
    best_blocks = np.array([], dtype=float)

    for trial in range(max_tries):
        out_meta["tries"] = trial + 1
        if w >= n:
            break

        time_values = ds._time_values_for_series(series)
        bm_series = ds._process_column(series, w, method, time_values=time_values)
        if bm_series is None or len(bm_series) == 0:
            w += 1
            continue

        blocks = np.asarray(bm_series.values, dtype=float)
        n_blocks = int(blocks.size)

        if n_blocks < max(3, min_blocks):
            if best_blocks.size == 0:
                best_blocks = blocks
                out_meta["best_window"] = int(w)
            out_meta["warnings"].append("too_few_blocks_for_ljungbox")
            break

        lags = lb_lags if lb_lags is not None else max(1, min(20, n_blocks // 4))
        try:
            lb = acorr_ljungbox(blocks, lags=[lags], return_df=True)
            pval = float(lb["lb_pvalue"].iloc[0])
        except Exception:  # noqa: BLE001
            pval = np.nan

        if np.isfinite(pval) and pval > float(out_meta["best_pvalue"]):
            out_meta["best_pvalue"] = pval
            out_meta["best_window"] = int(w)
            best_blocks = blocks

        if np.isfinite(pval) and pval >= lb_alpha:
            out_meta.update(
                {
                    "status": "independent",
                    "window_used": int(w),
                    "n_blocks": int(n_blocks),
                    "ljungbox_pvalue": float(pval),
                    "ljungbox_lags": int(lags),
                }
            )
            return blocks, out_meta

        w += 1

    if best_blocks.size > 0 and out_meta["best_window"] is not None:
        n_blocks = int(best_blocks.size)
        lags = lb_lags if lb_lags is not None else max(1, min(20, n_blocks // 4))
        try:
            lb = acorr_ljungbox(best_blocks, lags=[lags], return_df=True)
            pval = float(lb["lb_pvalue"].iloc[0])
        except Exception:  # noqa: BLE001
            pval = np.nan
        out_meta.update(
            {
                "status": "best_p",
                "window_used": int(out_meta["best_window"]),
                "n_blocks": int(n_blocks),
                "ljungbox_pvalue": float(pval) if np.isfinite(pval) else np.nan,
                "ljungbox_lags": int(lags),
            }
        )
        out_meta["warnings"].append("independence_failed_used_best_p_window")
        return best_blocks, out_meta

    out_meta["status"] = "best_p"
    out_meta["warnings"].append("no_valid_blocks")
    return np.array([], dtype=float), out_meta


def pool_block_means(
    data_streams: List[DataStream],
    col: str,
    window_size: Optional[int] = None,
    method: str = "non-overlapping",
    lb_alpha: float = 0.05,
    lb_lags: Optional[int] = None,
    max_tries: int = 25,
    min_blocks: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    For all members: autotune per-member window size, then pool block means.

    Parameters
    ----------
    data_streams : list of DataStream
    col : str
    window_size : int or None
    method : str
    lb_alpha : float
    lb_lags : int or None
    max_tries : int
    min_blocks : int

    Returns
    -------
    (pooled_blocks, meta)
    """
    pooled: List[np.ndarray] = []
    meta: Dict[str, Any] = {
        "members_used": 0,
        "member_windows": [],
        "member_blocks": [],
        "member_status": [],
        "member_pvalues": [],
        "member_lags": [],
        "member_warnings": [],
    }

    for ds in data_streams:
        blocks, m = autotune_member_blocks_until_independent(
            ds=ds,
            col=col,
            window_size=window_size,
            method=method,
            lb_alpha=lb_alpha,
            lb_lags=lb_lags,
            max_tries=max_tries,
            min_blocks=min_blocks,
        )
        if blocks is None or blocks.size == 0:
            continue

        pooled.append(blocks.astype(float, copy=False))
        meta["members_used"] += 1
        meta["member_windows"].append(int(m.get("window_used") or -1))
        meta["member_blocks"].append(int(m.get("n_blocks", 0) or 0))
        meta["member_status"].append(m.get("status") or "unknown")
        meta["member_pvalues"].append(float(m.get("ljungbox_pvalue", np.nan)))
        meta["member_lags"].append(int(m.get("ljungbox_lags", 0) or 0))
        meta["member_warnings"].append(m.get("warnings", []))

    if not pooled:
        return np.array([], dtype=float), meta

    return np.concatenate(pooled, axis=0), meta


# ---------------------------------------------------------------------------
# Technique 0 — average-ensemble statistics
# ---------------------------------------------------------------------------

def tech0_stats_for_col(
    data_streams: List[DataStream],
    col: str,
    ddof: int = 1,
    method: str = "non-overlapping",
    window_size: Optional[int] = None,
    avg_ds: Optional[DataStream] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute Technique-0 statistics for one column.

    Uses a pre-computed average-ensemble DataStream (or builds it from
    *data_streams* if *avg_ds* is ``None``). Runs
    :meth:`DataStream.compute_statistics` on the single averaged trace.

    Parameters
    ----------
    data_streams : list of DataStream
        Used to build the average when *avg_ds* is ``None``.
    col : str
    ddof : int
    method : str
    window_size : int or None
    avg_ds : DataStream or None
        Pre-computed ensemble average.

    Returns
    -------
    (stat_dict, meta_dict)
    """
    from quends.base.ensemble_utils import compute_average_ensemble  # noqa: PLC0415

    if avg_ds is None:
        avg_ds = compute_average_ensemble(data_streams)

    if col not in avg_ds.data.columns:
        out: Dict[str, Any] = {
            "mean": np.nan,
            "mean_uncertainty": np.nan,
            "confidence_interval": (np.nan, np.nan),
            "pm_std": (np.nan, np.nan),
            "warning": f"Column '{col}' missing from average ensemble.",
        }
        return out, {}

    s = avg_ds.compute_statistics(
        column_name=col, ddof=ddof, method=method, window_size=window_size
    )
    stat = s.get(col, {}) if isinstance(s, dict) else {}
    return stat, {"n_members_averaged": len(data_streams)}


# ---------------------------------------------------------------------------
# Technique 1 — pooled-block statistics
# ---------------------------------------------------------------------------

def tech1_pooled_stats_for_col(
    data_streams: List[DataStream],
    col: str,
    ddof: int = 1,
    window_size: Optional[int] = None,
    method: str = "non-overlapping",
    lb_lags: Optional[int] = None,
    lb_alpha: float = 0.05,
    pooled_lb_alpha_bad: float = 0.01,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute Technique-1 (pooled-block) statistics for one column.

    Policy:

    * Always compute ``sem_n`` (sd/sqrt(n_blocks)) and ``sem_ess``
      (sd/sqrt(ESS_blocks)).
    * Default reported SEM = ``sem_n``.
    * If pooled Ljung-Box p-value < *pooled_lb_alpha_bad*, switch to
      ``sem_ess``.

    Parameters
    ----------
    data_streams : list of DataStream
    col : str
    ddof : int
    window_size : int or None
    method : str
    lb_lags : int or None
    lb_alpha : float
    pooled_lb_alpha_bad : float

    Returns
    -------
    (stat_dict, meta_dict)
    """
    pooled, meta = pool_block_means(
        data_streams=data_streams,
        col=col,
        window_size=window_size,
        method="non-overlapping",
        lb_alpha=lb_alpha,
        lb_lags=lb_lags,
    )

    member_status = meta.get("member_status", [])
    all_independent = bool(member_status) and all(
        s == "independent" for s in member_status
    )
    some_best_p = any(s == "best_p" for s in member_status)

    out: Dict[str, Any] = {}
    if pooled.size == 0:
        out.update(
            {
                "mean": np.nan,
                "variance": np.nan,
                "ess_blocks": np.nan,
                "n_short_averages": 0,
                "mean_uncertainty": np.nan,
                "mean_uncertainty_sem_n": np.nan,
                "mean_uncertainty_sem_ess": np.nan,
                "se_method": None,
                "warning": "No pooled block means available.",
                "confidence_interval": (np.nan, np.nan),
                "pm_std": (np.nan, np.nan),
                "independent": None,
                "ljungbox_pvalue": np.nan,
                "ljungbox_lags": 0,
                "member_all_independent": all_independent,
                "member_some_best_p": some_best_p,
            }
        )
        meta["pooled_blocks"] = 0
        return out, meta

    n_blocks = int(pooled.size)
    mu = float(np.mean(pooled))
    var_blocks = float(np.var(pooled, ddof=ddof)) if n_blocks > 1 else 0.0
    ess_blocks = float(max(1.0, _geyer_ess_on_blocks(pooled)))
    sem_n = float(np.sqrt(var_blocks) / np.sqrt(max(1.0, float(n_blocks))))
    sem_ess = float(np.sqrt(var_blocks) / np.sqrt(ess_blocks))

    lags = lb_lags if lb_lags is not None else int(max(1, min(20, n_blocks // 4)))
    try:
        lb = acorr_ljungbox(pooled, lags=[lags], return_df=True)
        pooled_pval = float(lb["lb_pvalue"].iloc[0])
        pooled_independent = bool(pooled_pval >= lb_alpha)
    except Exception:  # noqa: BLE001
        pooled_pval = np.nan
        pooled_independent = None

    sem_reported = sem_n
    se_method = "sem_n"
    warning: Optional[str] = "Some members used best_p window." if some_best_p else None

    if np.isfinite(pooled_pval) and pooled_pval < float(pooled_lb_alpha_bad):
        sem_reported = sem_ess
        se_method = "sem_ess (pooled_LB_bad)"
        extra = (
            f"Pooled LB p={pooled_pval:.3g} < {pooled_lb_alpha_bad}; "
            "switched to ESS-based SEM."
        )
        warning = extra if warning is None else (warning + " " + extra)

    ci = (float(mu - 1.96 * sem_reported), float(mu + 1.96 * sem_reported))
    pm_std = (float(mu - sem_reported), float(mu + sem_reported))

    out.update(
        {
            "mean": mu,
            "variance": var_blocks,
            "ess_blocks": ess_blocks,
            "n_short_averages": n_blocks,
            "mean_uncertainty": sem_reported,
            "mean_uncertainty_sem_n": sem_n,
            "mean_uncertainty_sem_ess": sem_ess,
            "se_method": se_method,
            "warning": warning,
            "confidence_interval": ci,
            "pm_std": pm_std,
            "independent": pooled_independent,
            "ljungbox_pvalue": pooled_pval,
            "ljungbox_lags": int(lags),
            "member_all_independent": all_independent,
            "member_some_best_p": some_best_p,
        }
    )
    meta["pooled_blocks"] = n_blocks
    return out, meta


# ---------------------------------------------------------------------------
# Technique 2 — member-wise then inverse-variance aggregate
# ---------------------------------------------------------------------------

def tech2_stats_for_col(
    data_streams: List[DataStream],
    col: str,
    ddof: int = 1,
    method: str = "non-overlapping",
    window_size: Optional[int] = None,
    diagnostics: str = "compact",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute Technique-2 statistics for one column.

    Calls :meth:`DataStream.compute_statistics` on each member and aggregates
    via inverse-variance weighting (fallback: simple mean when no variances are
    available).

    Parameters
    ----------
    data_streams : list of DataStream
    col : str
    ddof : int
    method : str
    window_size : int or None
    diagnostics : {"compact", "full"}
        ``"full"`` includes per-member statistics in the output.

    Returns
    -------
    (stat_dict, meta_dict)
    """
    indiv: Dict[str, Any] = {}
    means: List[float] = []
    ses: List[float] = []
    w_means: List[float] = []
    weights: List[float] = []
    var_list: List[float] = []
    ess_list: List[float] = []
    nsa_list: List[float] = []

    for i, ds in enumerate(data_streams):
        key = f"Member {i}"
        if col not in ds.data.columns:
            continue
        s = ds.compute_statistics(
            column_name=col, ddof=ddof, method=method, window_size=window_size
        )
        stat = s.get(col, {}) if isinstance(s, dict) else {}
        if not isinstance(stat, dict):
            continue

        mu = stat.get("mean", np.nan)
        se = stat.get("mean_uncertainty", np.nan)
        ci = stat.get("confidence_interval", (np.nan, np.nan))
        pm = stat.get("pm_std", (np.nan, np.nan))
        ess_i = stat.get("ess_blocks", stat.get("block_effective_n", np.nan))
        nsa_i = stat.get("n_short_averages", np.nan)
        var_i = stat.get("variance", np.nan)

        indiv[key] = {
            "mean": mu,
            "mean_uncertainty": se,
            "variance": var_i,
            "confidence_interval": ci,
            "pm_std": pm,
            "ess_blocks": ess_i,
            "n_short_averages": nsa_i,
        }
        if np.isfinite(mu):
            means.append(float(mu))
        if np.isfinite(se):
            ses.append(float(se))
        if np.isfinite(mu) and np.isfinite(se) and se > 0:
            w_means.append(float(mu))
            weights.append(1.0 / float(se) ** 2)
        if np.isfinite(var_i):
            var_list.append(float(var_i))
        if np.isfinite(ess_i):
            ess_list.append(float(ess_i))
        if np.isfinite(nsa_i):
            nsa_list.append(float(nsa_i))

    if weights and w_means:
        w = np.asarray(weights, dtype=float)
        m = np.asarray(w_means, dtype=float)
        w_sum = float(np.sum(w))
        mu_ens = float(np.sum(w * m) / w_sum) if w_sum > 0 else np.nan
        se_ens = float(np.sqrt(1.0 / w_sum)) if w_sum > 0 else np.nan
    else:
        mu_ens = float(np.nanmean(means)) if means else np.nan
        se_ens = float(np.nanmean(ses)) if ses else np.nan

    var_ens = float(np.nanmean(var_list)) if var_list else np.nan
    ess_ens = float(np.nanmean(ess_list)) if ess_list else np.nan
    nsa_ens = float(np.nanmean(nsa_list)) if nsa_list else np.nan
    ci_ens = (
        float(mu_ens - 1.96 * se_ens) if np.isfinite(se_ens) else np.nan,
        float(mu_ens + 1.96 * se_ens) if np.isfinite(se_ens) else np.nan,
    )

    out: Dict[str, Any] = {
        "mean": mu_ens,
        "mean_uncertainty": se_ens,
        "confidence_interval": ci_ens,
        "pm_std": (
            (mu_ens - se_ens)
            if np.isfinite(mu_ens) and np.isfinite(se_ens)
            else np.nan,
            (mu_ens + se_ens)
            if np.isfinite(mu_ens) and np.isfinite(se_ens)
            else np.nan,
        ),
        "variance": var_ens,
        "ess_blocks": ess_ens,
        "n_short_averages": nsa_ens,
        "se_method": "tech2_inverse_variance_weighted",
        "warning": None,
        "individual": indiv if diagnostics == "full" else None,
    }
    meta: Dict[str, Any] = {"n_members_used": len(indiv)}
    return out, meta


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def compute_ensemble_statistics(
    data_streams: List[DataStream],
    column_name: Optional[Any] = None,
    ddof: int = 1,
    method: str = "non-overlapping",
    window_size: Optional[int] = None,
    technique: int = 1,
    diagnostics: str = "compact",
) -> Dict:
    """
    Aggregate mean, SEM, CI, ±SEM, variance, and ESS across an ensemble.

    Dispatches to the appropriate technique helper.

    Parameters
    ----------
    data_streams : list of DataStream
    column_name : str, list of str, or None
        ``None`` → all common variables.
    ddof : int
    method : {"non-overlapping", "sliding"}
    window_size : int or None
    technique : {0, 1, 2}
        0 — average-ensemble; 1 — pooled-block; 2 — member-wise IVW.
    diagnostics : {"compact", "full"}

    Returns
    -------
    dict
        ``{"results": {col: {stats}}, "metadata": {...}}``

    Raises
    ------
    ValueError
        If *technique* is not 0, 1, or 2.
    """
    from quends.base.ensemble_utils import (  # noqa: PLC0415
        compute_average_ensemble,
        resolve_cols,
    )

    cols = resolve_cols(data_streams, column_name)
    stats: Dict = {}
    metadata: Dict = {}

    if technique == 0:
        avg_ds = compute_average_ensemble(data_streams)
        meta_cols: Dict = {}
        for col in cols:
            col_stats, col_meta = tech0_stats_for_col(
                data_streams=data_streams,
                col=col,
                ddof=ddof,
                method=method,
                window_size=window_size,
                avg_ds=avg_ds,
            )
            stats[col] = col_stats
            meta_cols[col] = col_meta
        metadata["technique_0_average_ensemble"] = meta_cols
        return {"results": stats, "metadata": metadata}

    elif technique == 1:
        meta_cols = {}
        for col in cols:
            col_stats, col_meta = tech1_pooled_stats_for_col(
                data_streams=data_streams,
                col=col,
                ddof=ddof,
                window_size=window_size,
                method=method,
            )
            stats[col] = col_stats
            meta_cols[col] = col_meta
        metadata["technique_1_pooled"] = meta_cols
        return {"results": stats, "metadata": metadata}

    elif technique == 2:
        meta_cols = {}
        for col in cols:
            col_stats, col_meta = tech2_stats_for_col(
                data_streams=data_streams,
                col=col,
                ddof=ddof,
                method=method,
                window_size=window_size,
                diagnostics=diagnostics,
            )
            stats[col] = {
                "mean": col_stats["mean"],
                "mean_uncertainty": col_stats["mean_uncertainty"],
                "confidence_interval": col_stats["confidence_interval"],
                "pm_std": col_stats["pm_std"],
                "variance": col_stats.get("variance", np.nan),
                "ess_blocks": col_stats.get("ess_blocks", np.nan),
                "n_short_averages": col_stats.get("n_short_averages", np.nan),
                "se_method": col_stats.get("se_method"),
                "warning": col_stats.get("warning"),
                "individual": col_stats.get("individual")
                if diagnostics == "full"
                else None,
            }
            meta_cols[col] = col_meta
        metadata["technique_2_memberwise"] = meta_cols
        return {"results": stats, "metadata": metadata}

    else:
        raise ValueError(
            f"Invalid technique {technique!r}. Use 0 (average-ensemble), "
            "1 (pooled-block), or 2 (member-wise IVW)."
        )
