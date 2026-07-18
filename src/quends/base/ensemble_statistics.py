"""
ensemble_statistics.py
----------------------
Module-level statistical functions for ensemble analysis.

These functions implement the three ensemble-statistics techniques and operate
on plain lists of :class:`~quends.base.data_stream.DataStream` objects so that
workflow classes and the :class:`~quends.base.ensemble.Ensemble` class share
the same computation path without code duplication.

Technique names
---------------
The three techniques are addressed by descriptive canonical names; legacy
``technique=0|1|2`` and ``"technique0"|"technique1"|"technique2"`` strings
are accepted for backward compatibility but normalised internally.

* ``"ensemble_average"`` — average traces element-wise into a single trace and
  run DataStream statistics on it (technique 0).
* ``"pooled_block_means"`` — autotune block window per member until block means
  pass Ljung-Box independence, pool blocks across members, compute statistics
  on the pooled series (technique 1; preferred for trimmed ensembles).
* ``"ivw_member_means"`` — call ``DataStream.compute_statistics()`` on each
  member, combine the per-member ``(mean, SE)`` pairs using inverse-variance
  weighting (technique 2; fallback to simple mean when SEs are unusable).

Public API
----------
pool_block_means
ensemble_average_stats_for_col
pooled_block_means_stats_for_col
ivw_member_means_stats_for_col
compute_ensemble_statistics
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .data_stream import DataStream
from .utils import _geyer_ess_on_blocks, confidence_multiplier, pooled_tau_int


# ---------------------------------------------------------------------------
# Technique-name normalisation
# ---------------------------------------------------------------------------

#: Canonical technique name strings.
ENSEMBLE_AVERAGE = "ensemble_average"
POOLED_BLOCK_MEANS = "pooled_block_means"
IVW_MEMBER_MEANS = "ivw_member_means"

#: Mapping of every accepted alias → canonical name.  Legacy integers and
#: ``"technique0"|"technique1"|"technique2"`` strings remain valid inputs;
#: internal code always uses the canonical strings above.
_TECHNIQUE_ALIASES: Dict[Any, str] = {
    # Technique 0
    0: ENSEMBLE_AVERAGE,
    "0": ENSEMBLE_AVERAGE,
    "technique0": ENSEMBLE_AVERAGE,
    "t0": ENSEMBLE_AVERAGE,
    "average": ENSEMBLE_AVERAGE,
    "ensemble_average": ENSEMBLE_AVERAGE,
    # Technique 1
    1: POOLED_BLOCK_MEANS,
    "1": POOLED_BLOCK_MEANS,
    "technique1": POOLED_BLOCK_MEANS,
    "t1": POOLED_BLOCK_MEANS,
    "pooled_block_means": POOLED_BLOCK_MEANS,
    "pool": POOLED_BLOCK_MEANS,
    # Technique 2
    2: IVW_MEMBER_MEANS,
    "2": IVW_MEMBER_MEANS,
    "technique2": IVW_MEMBER_MEANS,
    "t2": IVW_MEMBER_MEANS,
    "ivw_member_means": IVW_MEMBER_MEANS,
    "ivw": IVW_MEMBER_MEANS,
}


def _normalize_technique(technique: Union[int, str]) -> str:
    """
    Normalise any accepted technique alias to its canonical string name.

    Raises
    ------
    ValueError
        If *technique* is not a recognised alias.
    """
    key = technique
    if isinstance(key, str):
        key = key.strip().lower()
    if key in _TECHNIQUE_ALIASES:
        return _TECHNIQUE_ALIASES[key]
    raise ValueError(
        f"Invalid technique {technique!r}. Use one of "
        f"{sorted(set(_TECHNIQUE_ALIASES.values()))!r} "
        "(or legacy 0/1/2, 'technique0'/'technique1'/'technique2')."
    )


# ---------------------------------------------------------------------------
# Technique 1 — helpers
# ---------------------------------------------------------------------------

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

    Calls ``DataStream._process_column`` directly for each member — the single
    autotune entry point — so there is exactly **one** autotune loop per member.

    Parameters
    ----------
    data_streams : list of DataStream
    col : str
    window_size : int or None
        Starting window hint; auto-estimated via tau_int when ``None``.
    method : str
    lb_alpha : float
    lb_lags : int or None
        Single-lag override for Ljung-Box (backward compat).
        When ``None``, the canonical ``lag_set=(5, 10)`` is used.
    max_tries : int
        Maximum autotune iterations forwarded to ``autotune_blocks``.
    min_blocks : int
        Hard stop forwarded to ``autotune_blocks``.

    Returns
    -------
    (pooled_blocks, meta)
    """
    pooled: List[np.ndarray] = []
    meta: Dict[str, Any] = {
        "members_used": 0,
        "member_windows": [],
        "member_blocks": [],
        "member_ess": [],
        "member_status": [],
        "member_pvalues": [],
        "member_lags": [],
        "member_warnings": [],
    }

    lag_set = (int(lb_lags),) if lb_lags is not None else (5, 10)

    for ds in data_streams:
        if col not in ds.data.columns:
            continue
        series = ds.data[col].dropna()
        if series.size < 4:
            continue

        # Single autotune per member via the canonical helper.
        _, ab = ds._process_column(
            series,
            estimated_window=window_size,
            method=method,
            alpha=lb_alpha,
            lag_set=lag_set,
            B_min=min_blocks,
            min_blocks=min_blocks,
            max_iter=max_tries,
            w_min=2,
            c0=2.0,
        )
        blocks = ab["blocks"]
        if blocks.size == 0:
            continue

        pvals = ab["ljungbox_pvalues"]
        lags_used = ab["ljungbox_lags"]

        pooled.append(blocks.astype(float, copy=False))
        meta["members_used"] += 1
        meta["member_windows"].append(int(ab["window_size"]))
        meta["member_blocks"].append(int(ab["n_blocks"]))
        # Per-member effective sample size on that member's OWN block means
        # (no cross-member boundary — see AUDIT_REPORT H3).
        meta["member_ess"].append(float(max(1.0, _geyer_ess_on_blocks(blocks))))
        meta["member_status"].append(ab["independence_status"])
        meta["member_pvalues"].append(float(min(pvals)) if pvals else float("nan"))
        meta["member_lags"].append(int(lags_used[0]) if lags_used else 0)
        meta["member_warnings"].append([ab["warning"]] if ab.get("warning") else [])

    if not pooled:
        return np.array([], dtype=float), meta

    return np.concatenate(pooled, axis=0), meta


# ---------------------------------------------------------------------------
# Technique 0 — average-ensemble statistics
# ---------------------------------------------------------------------------

def ensemble_average_stats_for_col(
    data_streams: List[DataStream],
    col: str,
    ddof: int = 1,
    method: str = "non-overlapping",
    window_size: Optional[int] = None,
    avg_ds: Optional[DataStream] = None,
    confidence_level: float = 0.95,
    ci_method: str = "normal",
    tau_mode: str = "pooled",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute *ensemble_average* (T0) statistics for one column.

    ``tau_mode`` (default ``"pooled"``) sets the block window for the averaged
    trace:

    * ``"pooled"``     - ``w = round(tau*)`` with ``tau*`` pooled across the member
      traces.  Averaging N independent members scales the autocovariance by 1/N
      but leaves the *normalised* ACF unchanged (``rho_ea = rho``), so the member
      correlation time IS the correct time-scale for the EA trace — and pooling M
      member ACFs resolves it far better than the single averaged trace can.
    * ``"per_member"`` - use the EA trace's own tau (the plain tau-window rule).

    An explicit *window_size* overrides both.

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

    # Window for the EA trace: pooled tau* by default (see docstring).
    w_eff, tau_star = window_size, None
    if w_eff is None and tau_mode == "pooled":
        traces = [
            ds.data[col].dropna().to_numpy(dtype=float)
            for ds in data_streams
            if col in ds.data.columns
        ]
        traces = [t for t in traces if t.size > 3]
        if len(traces) >= 2:
            tau_star = pooled_tau_int(traces)
            w_eff = max(1, int(round(tau_star)))
    elif tau_mode not in ("pooled", "per_member"):
        raise ValueError("tau_mode must be 'pooled' or 'per_member'")

    s = avg_ds.compute_statistics(
        column_name=col,
        ddof=ddof,
        method=method,
        window_size=w_eff,
        confidence_level=confidence_level,
        ci_method=ci_method,
    )
    stat = s.get(col, {}) if isinstance(s, dict) else {}
    return stat, {
        "n_members_averaged": len(data_streams),
        "tau_mode": tau_mode,
        "tau_star": tau_star,
    }


# ---------------------------------------------------------------------------
# Technique 1 — pooled-block statistics
# ---------------------------------------------------------------------------

def pooled_block_means_stats_for_col(
    data_streams: List[DataStream],
    col: str,
    ddof: int = 1,
    window_size: Optional[int] = None,
    method: str = "non-overlapping",
    lb_lags: Optional[int] = None,
    lb_alpha: float = 0.05,
    pooled_lb_alpha_bad: float = 0.01,
    confidence_level: float = 0.95,
    ci_method: str = "normal",
    pooled_ess_method: str = "concat",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute *pooled_block_means* (T1) statistics for one column.

    Pipeline:

    1. :func:`pool_block_means` calls ``DataStream._process_column`` directly for
       each member → :func:`~quends.base.utils.autotune_blocks` (one autotune per
       member, the single canonical helper). Per-member block ESS and independence
       are recorded there.
    2. ESS and independence are combined **per member** (independent members'
       effective counts add; the ensemble status is all/some/none-independent from
       the per-member verdicts). No test is run on the cross-member concatenation,
       which would inject spurious boundary autocorrelation (AUDIT_REPORT H3).
    3. SEM policy: ``sem_n`` (sd/√n_blocks) when every member is independent;
       otherwise ``sem_ess`` (sd/√ESS_blocks). ``lb_lags`` and
       ``pooled_lb_alpha_bad`` are retained for backward compatibility but no
       longer drive a pooled test.

    Parameters
    ----------
    data_streams : list of DataStream
    col : str
    ddof : int
    window_size : int or None
    method : str
    lb_lags : int or None
        Single-lag override for the pooled Ljung-Box test (backward compat).
        When ``None``, the canonical ``lag_set=(5, 10)`` is used.
    lb_alpha : float
    pooled_lb_alpha_bad : float

    Returns
    -------
    (stat_dict, meta_dict)
        ``stat_dict`` canonical keys:

        ``mean``, ``variance``, ``ess_blocks``, ``n_short_averages``,
        ``mean_uncertainty``, ``mean_uncertainty_sem_n``, ``mean_uncertainty_sem_ess``,
        ``se_method``, ``warning``, ``confidence_interval``, ``pm_std``,
        ``window_size`` (median member window, or ``None`` if no members),
        ``independent``, ``independence_status``,
        ``ljungbox_pvalue`` (scalar min, backward-compat),
        ``ljungbox_pvalues`` (list, normalised schema),
        ``ljungbox_lags`` (list, normalised schema),
        ``member_all_independent``, ``member_some_best_p``.
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

    # Aggregate member windows for a representative window_size in the output.
    mw_raw = meta.get("member_windows", [])
    mw = [int(w) for w in mw_raw if isinstance(w, (int, float)) and w > 0]
    member_window_sizes: List[int] = list(mw)
    if mw:
        window_size_agg: Optional[int] = int(round(float(np.median(mw))))
        window_size_min: Optional[int] = int(min(mw))
        window_size_max: Optional[int] = int(max(mw))
    else:
        window_size_agg = None
        window_size_min = None
        window_size_max = None

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
                # Window-size schema (B2)
                "window_size": window_size_agg,
                "member_window_sizes": member_window_sizes,
                "window_size_summary": window_size_agg,
                "window_size_summary_method": "median",
                "window_size_min": window_size_min,
                "window_size_max": window_size_max,
                "independent": None,
                "independence_status": "unknown",
                "ljungbox_pvalue": np.nan,
                "ljungbox_pvalues": [],
                "ljungbox_lags": [],
                "ci_method": "normal",
                "confidence_level": 0.95,
                "member_all_independent": all_independent,
                "member_some_best_p": some_best_p,
            }
        )
        meta["pooled_blocks"] = 0
        return out, meta

    n_blocks = int(pooled.size)
    mu = float(np.mean(pooled))
    var_blocks = float(np.var(pooled, ddof=ddof)) if n_blocks > 1 else 0.0
    sem_n = float(np.sqrt(var_blocks) / np.sqrt(max(1.0, float(n_blocks))))

    # ESS and independence are derived PER MEMBER and then combined — never from
    # the cross-member concatenation. Joining independent members end-to-end would
    # inject spurious autocorrelation at the boundaries, corrupting both a Geyer
    # ESS and a Ljung-Box test on the pooled array. See AUDIT_REPORT H3.
    # Independent members' effective sample counts add, so the pooled effective
    # block count is the sum of per-member block ESS.
    # Pooled effective block count.  Two supported rules:
    #   "concat" (default) - Geyer ESS on the concatenated pooled block means.
    #   "sum"              - add the per-member ESS (members are independent, so
    #                        their effective counts add; avoids the cross-member
    #                        boundary in the concatenated array, see AUDIT_REPORT H3).
    # On an AR(1) benchmark (M=16, w=tau) "concat" ran ~+2% conservative on the
    # SE while "sum" was unbiased; "concat" is the default by request.
    member_ess = [float(e) for e in meta.get("member_ess", []) if np.isfinite(e)]
    ess_sum = float(max(1.0, sum(member_ess))) if member_ess else float(max(1.0, n_blocks))
    ess_concat = float(max(1.0, _geyer_ess_on_blocks(pooled))) if n_blocks >= 2 else 1.0
    if pooled_ess_method == "concat":
        ess_blocks = ess_concat
    elif pooled_ess_method == "sum":
        ess_blocks = ess_sum
    else:
        raise ValueError("pooled_ess_method must be 'concat' or 'sum'")
    sem_ess = float(np.sqrt(var_blocks) / np.sqrt(ess_blocks))

    # Ensemble independence verdict from per-member statuses (consistent vocabulary
    # with ivw_member_means — see H5). lb_lags / pooled_lb_alpha_bad are retained in
    # the signature for backward compatibility but no longer drive a pooled test.
    member_pvals = [
        float(p) for p in meta.get("member_pvalues", [])
        if isinstance(p, float) and np.isfinite(p)
    ]
    n_members_t1 = len(member_status)
    n_indep_t1 = sum(1 for s in member_status if s == "independent")
    if n_members_t1 and n_indep_t1 == n_members_t1:
        pooled_status = "all_independent"
    elif n_indep_t1 > 0:
        pooled_status = "some_independent"
    elif n_members_t1:
        pooled_status = "none_independent"
    else:
        pooled_status = "unknown"
    pooled_independent = bool(n_members_t1 and n_indep_t1 == n_members_t1)
    pooled_pval = float(min(member_pvals)) if member_pvals else np.nan
    pooled_pvals: List[float] = member_pvals
    pooled_tested_lags: List[int] = list(meta.get("member_lags", []))

    # SEM policy: ALWAYS the ESS-corrected SEM (sum of per-member block ESS),
    # which credits residual within-member autocorrelation.  Formerly the
    # block-count SEM (sem_n) was used when every member passed Ljung-Box, but
    # that gate has low power and left the pooled SE optimistic; this mirrors the
    # DataStream ess_blocks change so per-member, EA, and SR tell one story.
    warning: Optional[str] = "Some members used best_p window." if some_best_p else None
    sem_reported = sem_ess
    se_method = "sem_ess"
    if not pooled_independent:
        extra = "Not all members passed independence; using ESS-based SEM."
        warning = extra if warning is None else (warning + " " + extra)

    # Confidence-interval multiplier — preserves historical 1.96 for the
    # default (normal, 0.95) case; t-quantile when ci_method='t' with dof set
    # by which SEM was reported (sem_n → n_blocks-1; sem_ess → ess_blocks-1).
    if ci_method == "t":
        if "sem_ess" in se_method:
            ci_dof = max(1, int(round(ess_blocks)) - 1)
        else:
            ci_dof = max(1, n_blocks - 1)
    else:
        ci_dof = None
    ci_mult = confidence_multiplier(
        confidence_level=confidence_level, method=ci_method, dof=ci_dof
    )

    ci = (float(mu - ci_mult * sem_reported), float(mu + ci_mult * sem_reported))
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
            # Window-size schema (B2)
            "window_size": window_size_agg,     # median member window; None if no members
            "member_window_sizes": member_window_sizes,
            "window_size_summary": window_size_agg,
            "window_size_summary_method": "median",
            "window_size_min": window_size_min,
            "window_size_max": window_size_max,
            # Independence diagnostics — normalised schema
            "independent": pooled_independent,
            "independence_status": pooled_status,
            "ljungbox_pvalue": pooled_pval,     # scalar (min over lags) — backward-compat
            "ljungbox_pvalues": pooled_pvals,   # list — normalised schema
            "ljungbox_lags": pooled_tested_lags,  # list — normalised schema
            # CI provenance
            "ci_method": ci_method,
            "confidence_level": float(confidence_level),
            "member_all_independent": all_independent,
            "member_some_best_p": some_best_p,
        }
    )
    meta["pooled_blocks"] = n_blocks
    meta["schema_version"] = "1.0"
    return out, meta


# ---------------------------------------------------------------------------
# Technique 2 — member-wise then inverse-variance aggregate
# ---------------------------------------------------------------------------

def ivw_member_means_stats_for_col(
    data_streams: List[DataStream],
    col: str,
    ddof: int = 1,
    method: str = "non-overlapping",
    window_size: Optional[int] = None,
    diagnostics: str = "compact",
    confidence_level: float = 0.95,
    ci_method: str = "normal",
    tau_mode: str = "pooled",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute *ivw_member_means* (T2) statistics for one column.

    Calls :meth:`DataStream.compute_statistics` on each member (which uses the
    canonical :func:`~quends.base.utils.autotune_blocks` helper) and aggregates
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
        ``"full"`` includes per-member statistics (including per-member window
        and independence diagnostics) in ``stat_dict["individual"]``.

    Returns
    -------
    (stat_dict, meta_dict)
        ``stat_dict`` canonical keys:

        ``mean``, ``mean_uncertainty``, ``confidence_interval``, ``pm_std``,
        ``variance``, ``ess_blocks``, ``n_short_averages``, ``se_method``,
        ``warning``,
        ``window_size`` (mean of per-member windows, or ``None``),
        ``independence_status`` (``"all_independent"``, ``"some_independent"``,
        ``"none_independent"``, or ``"unknown"``),
        ``independent`` (``True`` iff all members are independent),
        ``ljungbox_pvalue`` (mean of per-member min p-values),
        ``ljungbox_pvalues`` (list of per-member min p-values),
        ``individual`` (per-member dicts when *diagnostics* == ``"full"``).
    """
    indiv: Dict[str, Any] = {}
    means: List[float] = []
    ses: List[float] = []
    w_means: List[float] = []
    weights: List[float] = []
    var_list: List[float] = []
    ess_list: List[float] = []
    nsa_list: List[float] = []
    # Diagnostic aggregation
    window_list: List[int] = []
    independent_list: List[Optional[bool]] = []
    pvalue_list: List[float] = []
    # First non-empty per-member ljungbox_lags list, captured to expose at top
    # level (all members use the same canonical lag_set, so any one is fine).
    lags_repr: List[int] = []

    # Per-member block window.  Default "pooled": borrow tau* from the whole
    # ensemble.  A single short member resolves its own tau poorly (on the t1200
    # runs the per-member tau spans 22-156 where the truth is ~50, and a member
    # drawing tau=124 is left with only ~11 blocks / ESS~5), so pooling M member
    # ACFs gives a far more stable window.  "per_member" uses each member's OWN
    # tau instead - required for a parameter scan, where members genuinely differ
    # and pooling would force one wrong tau on all of them.
    if tau_mode not in ("pooled", "per_member"):
        raise ValueError("tau_mode must be 'pooled' or 'per_member'")
    w_member, tau_star = window_size, None
    if w_member is None and tau_mode == "pooled":
        _tr = [
            d.data[col].dropna().to_numpy(dtype=float)
            for d in data_streams
            if col in d.data.columns
        ]
        _tr = [t for t in _tr if t.size > 3]
        if len(_tr) >= 2:
            tau_star = pooled_tau_int(_tr)
            w_member = max(1, int(round(tau_star)))

    for i, ds in enumerate(data_streams):
        key = f"Member {i}"
        if col not in ds.data.columns:
            continue
        s = ds.compute_statistics(
            column_name=col, ddof=ddof, method=method, window_size=w_member
        )
        stat = s.get(col, {}) if isinstance(s, dict) else {}
        if not isinstance(stat, dict):
            continue

        mu = stat.get("mean", np.nan)
        se = stat.get("mean_uncertainty", np.nan)
        ci = stat.get("confidence_interval", (np.nan, np.nan))
        pm = stat.get("pm_std", (np.nan, np.nan))
        ess_i = stat.get("ess_blocks", np.nan)
        nsa_i = stat.get("n_short_averages", np.nan)
        var_i = stat.get("variance", np.nan)
        # Diagnostics from compute_statistics
        ws_i = stat.get("window_size", np.nan)
        indep_i: Optional[bool] = stat.get("independent", None)
        indep_status_i: str = stat.get("independence_status", "unknown")
        lb_pval_i = stat.get("ljungbox_pvalue", np.nan)
        lb_pvals_i = stat.get("ljungbox_pvalues", [])
        lb_lags_i = stat.get("ljungbox_lags", [])

        indiv[key] = {
            "mean": mu,
            "mean_uncertainty": se,
            "variance": var_i,
            "confidence_interval": ci,
            "pm_std": pm,
            "ess_blocks": ess_i,
            "n_short_averages": nsa_i,
            # Per-member diagnostic keys (always populated)
            "window_size": ws_i,
            "independence_status": indep_status_i,
            "independent": indep_i,
            "ljungbox_pvalue": lb_pval_i,
            "ljungbox_pvalues": lb_pvals_i,
            "ljungbox_lags": lb_lags_i,
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
        # Diagnostic aggregation
        if isinstance(ws_i, (int, float)) and np.isfinite(float(ws_i)):
            window_list.append(int(ws_i))
        independent_list.append(indep_i)
        if isinstance(lb_pval_i, float) and np.isfinite(lb_pval_i):
            pvalue_list.append(lb_pval_i)
        if not lags_repr and isinstance(lb_lags_i, list) and lb_lags_i:
            lags_repr = [int(v) for v in lb_lags_i]

    if weights and w_means:
        w = np.asarray(weights, dtype=float)
        m = np.asarray(w_means, dtype=float)
        w_sum = float(np.sum(w))
        mu_ens = float(np.sum(w * m) / w_sum) if w_sum > 0 else np.nan
        se_ens = float(np.sqrt(1.0 / w_sum)) if w_sum > 0 else np.nan
        se_method_ens = "ivw_member_means"
    else:
        # No usable inverse-variance weights. Fall back to an equal-weight mean
        # of the member means. For independent members the SE of that mean is the
        # valid combination SE — NOT the average of the member SEMs (which would
        # not shrink with N). Var(mean of means) = (1/N^2) Σ se_i^2, so
        # se = sqrt(Σ se_i^2) / N.  See AUDIT_REPORT H4.
        mu_ens = float(np.nanmean(means)) if means else np.nan
        if ses:
            se_arr = np.asarray(ses, dtype=float)
            n_se = int(se_arr.size)
            se_ens = float(np.sqrt(np.nansum(se_arr**2)) / n_se)
        else:
            se_ens = np.nan
        se_method_ens = "equal_weight_combined"

    var_ens = float(np.nanmean(var_list)) if var_list else np.nan
    # ess_blocks / n_short_averages are reported as ensemble TOTALS (sum across
    # members), so the same keys mean the same thing here as in pooled_block_means
    # (T1) — total effective / total block count across the ensemble. See H5.
    ess_ens = float(np.nansum(ess_list)) if ess_list else np.nan
    nsa_ens = float(np.nansum(nsa_list)) if nsa_list else np.nan

    # Aggregate window diagnostics — switched to median for robustness and
    # consistency with T1 (B2).  The mean-based summary previously reported
    # was sensitive to outlier member windows.
    if window_list:
        window_size_ens: Optional[int] = int(round(float(np.median(window_list))))
        window_size_min: Optional[int] = int(min(window_list))
        window_size_max: Optional[int] = int(max(window_list))
    else:
        window_size_ens = None
        window_size_min = None
        window_size_max = None
    member_window_sizes: List[int] = list(window_list)

    has_indep_info = bool(independent_list)
    all_indep = has_indep_info and all(b is True for b in independent_list)
    some_indep = any(b is True for b in independent_list)
    if all_indep:
        indep_status_ens = "all_independent"
    elif some_indep:
        indep_status_ens = "some_independent"
    elif has_indep_info:
        indep_status_ens = "none_independent"
    else:
        indep_status_ens = "unknown"
    avg_pvalue = float(np.mean(pvalue_list)) if pvalue_list else np.nan

    # CI multiplier — ivw_member_means uses normal by default.  The 't' method
    # requires a well-defined dof, which is awkward for an inverse-variance
    # weighted estimator combining members of differing dof.  We currently
    # support only ci_method='normal' here and raise for 't' to avoid silently
    # using a statistically questionable formula.
    if ci_method == "t":
        raise NotImplementedError(
            "ci_method='t' is not supported for the 'ivw_member_means' "
            "technique because the inverse-variance-weighted estimator "
            "combines per-member estimates with different degrees of freedom; "
            "use 'normal' or switch to 'pooled_block_means'."
        )
    ci_mult = confidence_multiplier(
        confidence_level=confidence_level, method=ci_method, dof=None
    )

    ci_ens = (
        float(mu_ens - ci_mult * se_ens) if np.isfinite(se_ens) else np.nan,
        float(mu_ens + ci_mult * se_ens) if np.isfinite(se_ens) else np.nan,
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
        "se_method": se_method_ens,
        "warning": None,
        # Window-size schema (B2)
        "window_size": window_size_ens,
        "member_window_sizes": member_window_sizes,
        "window_size_summary": window_size_ens,
        "window_size_summary_method": "median",
        "window_size_min": window_size_min,
        "window_size_max": window_size_max,
        # Independence diagnostics
        "independence_status": indep_status_ens,
        "independent": all_indep,
        "ljungbox_pvalue": avg_pvalue,       # mean of per-member min p-values
        "ljungbox_pvalues": pvalue_list,     # list of per-member min p-values
        "ljungbox_lags": lags_repr,          # list — first non-empty member's lags (B1)
        # CI provenance
        "ci_method": ci_method,
        "confidence_level": float(confidence_level),
        "individual": indiv if diagnostics == "full" else None,
    }
    meta: Dict[str, Any] = {"n_members_used": len(indiv), "schema_version": "1.0"}
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
    technique: Union[int, str] = POOLED_BLOCK_MEANS,
    diagnostics: str = "compact",
    confidence_level: float = 0.95,
    ci_method: str = "normal",
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
    technique : {"ensemble_average", "pooled_block_means", "ivw_member_means"}
        Canonical technique name.  Legacy values ``0``/``1``/``2`` and
        ``"technique0"``/``"technique1"``/``"technique2"`` are accepted as
        backward-compatible aliases.  Default ``"pooled_block_means"``.
    diagnostics : {"compact", "full"}

    Returns
    -------
    dict
        ``{"results": {col: {stats}}, "metadata": {...}}``

        Metadata key for each technique:

        * ``"technique_0_ensemble_average"``
        * ``"technique_1_pooled_block_means"``
        * ``"technique_2_ivw_member_means"``

    Raises
    ------
    ValueError
        If *technique* is not a recognised alias.
    """
    from quends.base.ensemble_utils import (  # noqa: PLC0415
        compute_average_ensemble,
        resolve_cols,
    )

    canonical = _normalize_technique(technique)
    cols = resolve_cols(data_streams, column_name)
    stats: Dict = {}
    # Common run-level metadata (read directly by budget/wall-clock studies).
    metadata: Dict = {
        "estimator": canonical,
        "n_members": len(data_streams),
        "total_samples": int(sum(len(ds.data) for ds in data_streams)),
        "schema_version": "1.0",
    }

    if canonical == ENSEMBLE_AVERAGE:
        avg_ds = compute_average_ensemble(data_streams)
        meta_cols: Dict = {}
        for col in cols:
            col_stats, col_meta = ensemble_average_stats_for_col(
                data_streams=data_streams,
                col=col,
                ddof=ddof,
                method=method,
                window_size=window_size,
                avg_ds=avg_ds,
                confidence_level=confidence_level,
                ci_method=ci_method,
            )
            stats[col] = col_stats
            meta_cols[col] = col_meta
        metadata["technique_0_ensemble_average"] = meta_cols
        return {"results": stats, "metadata": metadata}

    if canonical == POOLED_BLOCK_MEANS:
        meta_cols = {}
        for col in cols:
            col_stats, col_meta = pooled_block_means_stats_for_col(
                data_streams=data_streams,
                col=col,
                ddof=ddof,
                window_size=window_size,
                method=method,
                confidence_level=confidence_level,
                ci_method=ci_method,
            )
            stats[col] = col_stats
            meta_cols[col] = col_meta
        metadata["technique_1_pooled_block_means"] = meta_cols
        return {"results": stats, "metadata": metadata}

    # IVW_MEMBER_MEANS
    meta_cols = {}
    for col in cols:
        col_stats, col_meta = ivw_member_means_stats_for_col(
            data_streams=data_streams,
            col=col,
            ddof=ddof,
            method=method,
            window_size=window_size,
            diagnostics=diagnostics,
            confidence_level=confidence_level,
            ci_method=ci_method,
        )
        # Full pass-through so any new diagnostic keys added to
        # ivw_member_means_stats_for_col automatically appear here;
        # individual is suppressed when diagnostics != "full".
        stats[col] = dict(col_stats)
        if diagnostics != "full":
            stats[col]["individual"] = None
        meta_cols[col] = col_meta
    metadata["technique_2_ivw_member_means"] = meta_cols
    return {"results": stats, "metadata": metadata}
