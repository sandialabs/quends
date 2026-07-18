import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

SCHEMA_VERSION = "1.0"


class StatsResult(dict):
    """A ``{column: stats}`` mapping with an attached ``.metadata`` dict.

    Subclasses :class:`dict`, so it behaves exactly like the historical
    ``{column: {...}}`` return value — ``res[col]["mean"]``, ``res == plain_dict``,
    iteration, ``.get`` all work unchanged. It only *adds* a ``.metadata``
    attribute (run-level info such as estimator, sample counts, schema_version),
    so existing callers keep working while new code can read provenance.
    """

    def __init__(self, results=None, metadata=None):
        super().__init__(results or {})
        self.metadata = dict(metadata or {})

    def __repr__(self):  # keep dict repr but hint at the extra attribute
        return f"StatsResult({dict.__repr__(self)}, metadata={self.metadata!r})"


def power_law_model(n, A, p):
    return A / (n**p)


# ---------------------------------------------------------------------------
# Confidence-interval multiplier
# ---------------------------------------------------------------------------


def confidence_multiplier(
    confidence_level: float = 0.95,
    method: str = "normal",
    dof: Optional[int] = None,
) -> float:
    """
    Return the multiplier ``z`` such that ``CI = mean ± z × SE``.

    Parameters
    ----------
    confidence_level : float
        Two-sided confidence level in (0, 1).  Default ``0.95``.
    method : {"normal", "t"}
        ``"normal"`` — standard-normal quantile.
        ``"t"`` — Student's *t* quantile (requires *dof*).
    dof : int or None
        Degrees of freedom for the *t* distribution.  Required when
        ``method="t"``.  Ignored otherwise.

    Returns
    -------
    float
        The CI multiplier.

    Notes
    -----
    For the historical default ``(method="normal", confidence_level=0.95)``,
    this function returns the literal value ``1.96`` to preserve byte-for-byte
    backward compatibility with previously-stored results.  For all other
    parameter values, the quantile is computed exactly via :mod:`scipy.stats`.
    """
    if method == "normal":
        # Preserve the exact historical 1.96 value for the default case so
        # that older tests/results that hard-code it remain reproducible.
        if confidence_level == 0.95:
            return 1.96
        from scipy.stats import norm  # local import — avoid eager scipy load

        return float(norm.ppf((1.0 + float(confidence_level)) / 2.0))

    if method == "t":
        if dof is None or int(dof) < 1:
            raise ValueError(
                "ci_method='t' requires a positive integer dof; " f"got dof={dof!r}."
            )
        from scipy.stats import t as _t  # local import

        return float(_t.ppf((1.0 + float(confidence_level)) / 2.0, df=int(dof)))

    raise ValueError(f"Unknown ci_method {method!r}; choose 'normal' or 't'.")


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


def _resolve_columns(data, column_names):
    if column_names is None:
        return [col for col in data.columns if col != "time"]
    return [column_names] if isinstance(column_names, str) else column_names


def stationarity_results(result):
    """
    Normalize supported stationarity return schemas to a plain results mapping.

    DataStream currently returns ``{column: bool}``, while richer callers may
    use ``{"results": {column: bool}, "metadata": ...}``.  Keeping this small
    adapter at call sites prevents schema drift from turning into indexing bugs.
    """
    if isinstance(result, dict) and isinstance(result.get("results"), dict):
        return result["results"]
    if isinstance(result, dict):
        return result
    return {}


def stationarity_value(result, column, default=False):
    """Extract one column's stationarity boolean from any supported schema."""
    return stationarity_results(result).get(column, default)


def _geyer_ess_on_blocks(block_means: np.ndarray) -> float:
    """
    Geyer positive-pair ESS on an already block-meaned series.
    Stops accumulating pairs once rho_t + rho_{t+1} < 0.
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
    return max(1.0, n / (1.0 + 2.0 * s))


def _tau_int_geyer_from_acf(rho: np.ndarray) -> float:
    """
    Estimate integrated autocorrelation time tau_int via Geyer positive-pair truncation.
    rho[0] must equal 1 (standard ACF array).
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
    return float(max(1.0, 1.0 + 2.0 * s))


def _ljung_box_pass(
    block_means: np.ndarray,
    alpha: float = 0.05,
    lag_set=(5, 10),
) -> tuple:
    """
    Ljung-Box test on block means at multiple lags.
    Pass means p-value > alpha for ALL tested lags.

    Returns (passed: bool, details: dict).
    """
    bm = np.asarray(block_means, dtype=float)
    n_blocks = bm.size
    if n_blocks < 2:
        return False, {"n_blocks": int(n_blocks), "lags": [], "pvalues": []}

    tested_lags, pvalues = [], []
    _seen_lags: set = set()
    for lag in lag_set:
        L = int(min(lag, n_blocks - 1))
        if L < 1 or L in _seen_lags:
            continue
        _seen_lags.add(L)
        try:
            lb = acorr_ljungbox(bm, lags=[L], return_df=True)
            pvalues.append(float(lb["lb_pvalue"].iloc[0]))
            tested_lags.append(L)
        except Exception:
            pass

    if not tested_lags:
        return False, {"n_blocks": int(n_blocks), "lags": [], "pvalues": []}

    passed = all(p > alpha for p in pvalues)
    return passed, {"n_blocks": int(n_blocks), "lags": tested_lags, "pvalues": pvalues}


# ---------------------------------------------------------------------------
# Shared block-averaging and autotune helpers
# ---------------------------------------------------------------------------


def _compute_block_means(
    x: np.ndarray, w: int, method: str = "non-overlapping"
) -> np.ndarray:
    """
    Compute block means from a raw array without DataStream overhead.

    Parameters
    ----------
    x : np.ndarray, dtype float
        Raw series values (already dropna'd and converted to float).
    w : int
        Block / window size.
    method : {"non-overlapping", "sliding"}
        "non-overlapping" returns floor(n/w) non-overlapping block means.
        "sliding" returns a pandas rolling mean (NaN head dropped).

    Returns
    -------
    np.ndarray of float — the block-mean values.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if method == "non-overlapping":
        w = int(max(1, w))
        n_blocks = n // w
        if n_blocks < 1:
            return np.array([], dtype=float)
        return x[: n_blocks * w].reshape(n_blocks, w).mean(axis=1)
    elif method == "sliding":
        s = pd.Series(x).rolling(window=int(w)).mean().dropna()
        return s.values.astype(float)
    else:
        raise ValueError(
            f"method must be 'non-overlapping' or 'sliding'; got {method!r}"
        )


def _estimate_tau_int_from_series(x: np.ndarray) -> float:
    """
    Estimate the integrated autocorrelation time (tau_int) from a raw 1-D array.

    ``tau_int`` is used as the signal decorrelation length. It is measured in
    numbers of samples / array points, not physical time units.

    Uses Geyer positive-pair truncation of the sample ACF. Always returns a
    value >= 1.0. If the estimated decorrelation length is large compared to
    the maximum lag used to compute the ACF, a warning is emitted because the
    estimate may be under-resolved.

    Parameters
    ----------
    x : array-like
        1-D float array, already dropna'd.

    Returns
    -------
    float
        Estimated decorrelation length, in number of samples / points.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return 1.0
    nlags = max(1, min(n // 4, 2000))
    r = acf(x, nlags=nlags, fft=False)
    # decorrelation length
    tau_int = _tau_int_geyer_from_acf(r)

    # warn when decorrelation length is about same size as lag cutoff in autocorrelation function
    if tau_int >= 0.5 * nlags:
        warnings.warn(
            "The computed signal decorrelation time is large compared to the "
            "max lag in the computation of the autocorrelation. Results may "
            f"be inaccurate. Estimated tau_int={tau_int:.2f}, nlags={nlags}."
        )

    return tau_int


def autotune_blocks(
    x,
    window_size=None,
    method: str = "non-overlapping",
    alpha: float = 0.05,
    lag_set=(5, 10),
    B_min: int = 15,
    min_blocks: int = 2,
    max_iter: int = 25,
    w_min: int = 5,
    c0: float = 2.0,
) -> dict:
    """
    Shared core helper: autotune block window until block means are independent.

    This is the single implementation used by *both* ``DataStream.compute_statistics``
    and the ensemble statistics pipeline (Techniques 1 and 2).  All other
    window-autotune helpers in this package are thin wrappers around this function.

    Algorithm
    ---------
    If *window_size* is ``None`` (autotune path):

    1. Estimate tau_int from the raw-series ACF via Geyer positive-pair
       truncation.
    2. Seed the starting window: ``w0 = max(w_min, ceil(c0 * tau_int))``,
       soft-capped to ``n // B_min`` so that at least *B_min* blocks are
       available at the start of the search.
    3. Iterate: compute non-overlapping block means, run
       ``_ljung_box_pass(lag_set, alpha)`` (lags are capped to ``n_blocks-1``
       and deduplicated), advance ``w += 1``.
    4. Return on the first window that passes (status ``"independent"``).
    5. If the loop exhausts *max_iter* or the block count drops below
       *min_blocks*, return the window with the best LB p-value seen
       (status ``"best_p"``).
    6. If no valid blocks exist at all, return status ``"too_few_blocks"``.

    If *window_size* is provided (user-window path):

    * Use the given window directly without searching.
    * Still run the LB diagnostic for informational purposes.
    * Always set status ``"user_window"``; SE should be computed via Geyer ESS.

    Parameters
    ----------
    x : array-like
        Raw series values.  NaNs and Infs are removed before processing.
    window_size : int or None
        User-supplied window size.  ``None`` triggers autotuning.
    method : {"non-overlapping", "sliding"}
        Block type.  Independence testing is designed for
        ``"non-overlapping"``; Ljung-Box on sliding means may be misleading.
    alpha : float
        Ljung-Box significance level.  Default ``0.05``.
    lag_set : tuple of int
        Lags passed to ``_ljung_box_pass``.  Pass condition: ``p > alpha``
        for **all** tested lags.  Lags are capped to ``n_blocks - 1`` and
        deduplicated automatically.  Default ``(5, 10)``.
    B_min : int
        Soft starting-window cap.  The seed window is capped so that at
        least *B_min* blocks are available at the start of the search.
        Matches the ``B_min`` parameter of
        ``DataStream._autotune_window_size``.  Default ``15``.
    min_blocks : int
        Hard stop.  The loop terminates immediately (without a LB test) if
        the block count drops below *min_blocks*.  The best window found so
        far is returned.  Default ``2`` (matches the old DataStream
        behaviour of continuing until fewer than 2 blocks remain).
    max_iter : int
        Maximum window-increment iterations.  Default ``25``.
    w_min : int
        Minimum allowed window size.  Default ``5``.
    c0 : float
        Multiplier for tau_int: ``w0 = ceil(c0 * tau_int)``.  Default ``2.0``.

    Returns
    -------
    dict
        Keys:

        ``blocks`` : np.ndarray
            Final block-mean values (may be empty if no valid blocks).
        ``window_size`` : int
            Chosen window size.
        ``n_blocks`` : int
            Number of block means.
        ``independence_status`` : str
            ``"independent"``, ``"best_p"``, ``"user_window"``, or
            ``"too_few_blocks"``.
        ``independent`` : bool
            ``True`` iff status is ``"independent"``.
        ``ljungbox_lags`` : list[int]
            Lags actually tested at the chosen window.
        ``ljungbox_pvalues`` : list[float]
            P-values at each tested lag.
        ``best_pvalue`` : float
            Minimum p-value across lags at the chosen window, or NaN.
        ``tau_int`` : float
            Estimated tau_int (NaN for ``user_window`` path).
        ``initial_window`` : int
            Seed window before iteration starts.
        ``iterations`` : int
            Loop iterations used.
        ``autotuned`` : bool
            ``False`` for ``user_window`` path.
        ``warning`` : str or None
            Diagnostic message, or ``None`` when autotuning succeeded.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)

    # ------------------------------------------------------------------
    # User-supplied window path
    # ------------------------------------------------------------------
    if window_size is not None:
        w = int(window_size)
        blocks = _compute_block_means(x, w, method)
        n_blocks = int(blocks.size)
        lb: dict = {"lags": [], "pvalues": []}
        passed = False
        if n_blocks >= max(2, int(min_blocks)):
            passed, lb = _ljung_box_pass(blocks, alpha=alpha, lag_set=lag_set)
        pvals = lb.get("pvalues", [])
        return {
            "blocks": blocks,
            "window_size": int(w),
            "n_blocks": int(n_blocks),
            "independence_status": "user_window",
            "independent": bool(passed),
            "ljungbox_lags": lb.get("lags", []),
            "ljungbox_pvalues": pvals,
            "best_pvalue": float(min(pvals)) if pvals else float("nan"),
            "tau_int": float("nan"),
            "initial_window": int(w),
            "iterations": 0,
            "autotuned": False,
            "warning": "User-supplied window; SE computed via Geyer ESS on block means.",
        }

    # ------------------------------------------------------------------
    # Autotune path
    # ------------------------------------------------------------------
    if n < 2:
        return {
            "blocks": np.array([], dtype=float),
            "window_size": int(w_min),
            "n_blocks": 0,
            "independence_status": "too_few_blocks",
            "independent": False,
            "ljungbox_lags": [],
            "ljungbox_pvalues": [],
            "best_pvalue": float("nan"),
            "tau_int": 1.0,
            "initial_window": int(w_min),
            "iterations": 0,
            "autotuned": True,
            "warning": "Too few samples for block averaging.",
        }

    tau_int = _estimate_tau_int_from_series(x)

    # ---- Starting window ------------------------------------------------
    # Step 1: tau_int seed (no w_min floor here — handled below).
    w_tau = int(max(1, math.ceil(c0 * float(tau_int))))

    # Step 2: Soft cap — ensure >= B_min blocks at the start of the search.
    #   w_cap = floor(n / B_min), giving exactly B_min blocks at w_cap.
    w_cap = max(1, n // max(1, int(B_min)))

    # Step 3: Start at the *smaller* of tau seed and soft cap (so we always
    #   have at least B_min blocks).  Then apply the w_min advisory: raise w
    #   to w_min if the series is long enough to still satisfy B_min.
    #   Finally clamp to [1, n] so we always get >= 1 block.
    w = min(w_tau, w_cap)
    w = max(w, 1)
    # Apply w_min advisory without busting the n hard cap.
    w = min(max(w, min(int(w_min), n)), n)
    # -----------------------------------------------------------------------

    initial_window = int(w)
    best: dict = {
        "w": int(w),
        "blocks": np.array([], dtype=float),
        "p_min": float("-inf"),
        "lb": {"lags": [], "pvalues": []},
    }

    iteration_count = 0
    for _iter in range(int(max_iter)):
        iteration_count += 1

        blocks = _compute_block_means(x, w, method)
        n_blocks = int(blocks.size)

        if n_blocks < max(2, int(min_blocks)):
            # Too few blocks for a meaningful independence test.
            # Record the best blocks found (may only be 1 block) for the
            # fallback path; then stop searching.
            if best["blocks"].size == 0 and n_blocks > 0:
                best["w"] = int(w)
                best["blocks"] = blocks.copy()
            break

        passed, det = _ljung_box_pass(blocks, alpha=alpha, lag_set=lag_set)
        p_score = float(min(det["pvalues"])) if det.get("pvalues") else float("-inf")

        if p_score > best["p_min"]:
            best = {
                "w": int(w),
                "blocks": blocks.copy(),
                "p_min": p_score,
                "lb": det,
            }

        if passed:
            return {
                "blocks": blocks,
                "window_size": int(w),
                "n_blocks": int(n_blocks),
                "independence_status": "independent",
                "independent": True,
                "ljungbox_lags": det.get("lags", []),
                "ljungbox_pvalues": det.get("pvalues", []),
                "best_pvalue": float(p_score),
                "tau_int": float(tau_int),
                "initial_window": initial_window,
                "iterations": iteration_count,
                "autotuned": True,
                "warning": None,
            }

        w_next = w + 1
        # Stop when the next window would leave fewer than min_blocks blocks
        # OR fewer than 1 block (series exhausted).
        if w_next <= w or n // w_next < max(1, int(min_blocks)):
            break
        w = w_next

    # ------------------------------------------------------------------
    # Fallback — return best-p window found, or extreme fallback
    # ------------------------------------------------------------------
    if best["blocks"].size > 0:
        lb = best["lb"]
        p_score = best["p_min"]
        status = "best_p"
        # If we only have 1–(min_blocks-1) blocks and never ran a LB test,
        # call it "too_few_blocks" to signal low statistical power.
        if int(best["blocks"].size) < max(2, int(min_blocks)):
            status = "too_few_blocks"
        return {
            "blocks": best["blocks"],
            "window_size": int(best["w"]),
            "n_blocks": int(best["blocks"].size),
            "independence_status": status,
            "independent": False,
            "ljungbox_lags": lb.get("lags", []),
            "ljungbox_pvalues": lb.get("pvalues", []),
            "best_pvalue": (
                float(p_score) if np.isfinite(float(p_score)) else float("nan")
            ),
            "tau_int": float(tau_int),
            "initial_window": initial_window,
            "iterations": iteration_count,
            "autotuned": True,
            "warning": "Block means did not pass Ljung-Box; using best-p window.",
        }

    # Extreme fallback — use the smallest practical window (≤ n) so that
    # compute_statistics() can at least return a mean even for very short series.
    w_final = min(n, max(1, min(int(w_min), n)))
    blocks_final = _compute_block_means(x, w_final, method)
    return {
        "blocks": blocks_final,
        "window_size": int(w_final),
        "n_blocks": int(blocks_final.size),
        "independence_status": "too_few_blocks",
        "independent": False,
        "ljungbox_lags": [],
        "ljungbox_pvalues": [],
        "best_pvalue": float("nan"),
        "tau_int": float(tau_int),
        "initial_window": initial_window,
        "iterations": iteration_count,
        "autotuned": True,
        "warning": "Too few blocks for independence test; SE via Geyer ESS.",
    }


# ---------------------------------------------------------------------------
# tau-based window selection (replaces the Ljung-Box autotune)
# ---------------------------------------------------------------------------
def pooled_tau_int(traces, maxlag: Optional[int] = None) -> float:
    """
    Ensemble-pooled integrated autocorrelation time.

    Averages the *normalised* ACFs of the member traces and integrates the result
    with Geyer positive-pair truncation.  A single short trace estimates tau very
    noisily; averaging M of them is far better resolved (the same reason emcee
    pools the ACF across chains before integrating).

    ONLY valid for a HOMOGENEOUS ensemble (same case, different seeds/ICs).  For a
    parameter scan, where members genuinely differ, use per-member tau instead --
    pooling would impose one wrong correlation time on every member.

    Parameters
    ----------
    traces : sequence of 1-D array-like
        Member traces (internally truncated to their common length).
    maxlag : int, optional
        Defaults to ``nmin // 4``.

    Returns
    -------
    float
        Pooled tau_int (>= 1.0).
    """
    arrs = [np.asarray(t, dtype=float) for t in traces]
    arrs = [a[np.isfinite(a)] for a in arrs if np.size(a) > 3]
    if not arrs:
        return 1.0
    nmin = min(a.size for a in arrs)
    ml = int(maxlag or max(1, nmin // 4))

    def _nacf(x):
        x = x[:nmin] - x[:nmin].mean()
        m = x.size
        f = np.fft.rfft(x, 2 * m)
        ac = np.fft.irfft(f * np.conj(f))[:ml]
        return ac / ac[0] if ac[0] > 0 else ac

    rho = np.mean([_nacf(a) for a in arrs], axis=0)
    tau = 1.0
    for k in range(1, ml):
        if rho[k] <= 0:
            break
        tau += 2.0 * rho[k]
    return float(max(1.0, tau))


def tau_window_blocks(
    x,
    tau: Optional[float] = None,
    method: str = "non-overlapping",
    B_min_hard: int = 4,
    alpha: float = 0.05,
    lag_set=(5, 10),
) -> dict:
    """
    Block a series with ``w = round(tau)`` -- the tau-window rule.

    This replaces the Ljung-Box autotune as the default window rule.  Rationale:
    the SE is ``sd(block_means) / sqrt(ess_blocks)``, and that Geyer ESS already
    credits any residual between-block correlation.  Because the correction is
    applied, the blocks do **not** need to be independent, so there is no reason
    to keep widening the window until a low-power Ljung-Box test happens to pass.
    Keeping ``w`` small also leaves more blocks, which resolves both
    ``sd(block_means)`` and the block-level ESS better.

    The exact identity ``T_n = T_w * T_B`` (raw-series variance inflation factors
    into within-block x between-block) guarantees the resulting SE is invariant to
    ``w`` in expectation: widening the window moves correlation out of the
    between-block factor and into the within-block factor, leaving the product --
    and hence the SE -- unchanged.  So ``w`` is a precision knob, not a bias knob.

    ``tau`` may be supplied (e.g. an ensemble-pooled tau*); otherwise it is
    estimated from this series via Geyer positive-pair truncation.

    A hard floor keeps at least *B_min_hard* blocks.  It is a pathology guard for
    very short series / very large tau, not a tuning knob, and it is reported via
    ``warning`` when it binds.

    Returns a dict with the same keys as :func:`autotune_blocks`.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return {
            "blocks": np.array([], dtype=float),
            "window_size": 1,
            "n_blocks": 0,
            "independence_status": "too_few_blocks",
            "independent": False,
            "ljungbox_lags": [],
            "ljungbox_pvalues": [],
            "best_pvalue": float("nan"),
            "tau_int": 1.0,
            "initial_window": 1,
            "iterations": 0,
            "autotuned": False,
            "warning": "Too few samples for block averaging.",
        }

    tau_val = float(tau) if tau is not None else _estimate_tau_int_from_series(x)
    w_tau = max(1, int(round(tau_val)))
    w_cap = max(1, n // max(2, int(B_min_hard)))  # pathology guard only
    w = min(w_tau, w_cap)
    clamped = w < w_tau

    blocks = _compute_block_means(x, w, method)
    n_blocks = int(blocks.size)

    lb = {"lags": [], "pvalues": []}
    passed = False
    if n_blocks >= 2:  # diagnostic only; does NOT select w
        passed, lb = _ljung_box_pass(blocks, alpha=alpha, lag_set=lag_set)
    pvals = lb.get("pvalues", [])

    warn = None
    if clamped:
        warn = (
            f"tau-window clamped: w={w_tau} (tau={tau_val:.1f}) would leave "
            f"< {B_min_hard} blocks; using w={w}."
        )
    return {
        "blocks": blocks,
        "window_size": int(w),
        "n_blocks": n_blocks,
        "independence_status": "tau_window",
        "independent": bool(passed),
        "ljungbox_lags": lb.get("lags", []),
        "ljungbox_pvalues": pvals,
        "best_pvalue": float(min(pvals)) if pvals else float("nan"),
        "tau_int": float(tau_val),
        "initial_window": int(w_tau),
        "iterations": 0,
        "autotuned": False,
        "warning": warn,
    }


def _compute_ess(data, col, alpha=0.05):
    """
    Estimate the effective sample size (ESS) for one column of a DataFrame.

    Uses Geyer positive-pair truncation of the sample ACF — the same estimator
    used throughout the rest of QUENDS — rather than an absolute-value
    significance-threshold sum.

    Algorithm
    ---------
    1. Estimate ``tau_int`` via :func:`_estimate_tau_int_from_series`, which
       applies Geyer's positive-pair truncation to the raw-series ACF.
    2. Return ``ceil(n / tau_int)``, clamped to ``[1, n]``.

    Why this is better than the old ``abs(ACF) > threshold`` approach
    ------------------------------------------------------------------
    * **Sign handling:** Negative autocorrelations *increase* ESS (anti-correlated
      draws are more informative than i.i.d. draws).  The old code took absolute
      values, which made negative ACF reduce ESS — exactly wrong.
    * **Noisy long lags:** The significance threshold kept all lags whose |ACF|
      exceeded ``z / sqrt(n)``, including spuriously large values at long lags.
      Geyer truncation stops at the first pair whose sum is non-positive, so
      long-lag noise never accumulates.
    * **Consistency:** This estimator now agrees with the tau_int seeding used
      in :func:`autotune_blocks` and the ``_geyer_ess_on_blocks`` denominator.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the column.
    col : str
        Column name to estimate ESS for.
    alpha : float, optional
        Retained for API backward-compatibility; no longer used in the
        computation.  Default ``0.05``.

    Returns
    -------
    int
        Estimated ESS, or a ``dict`` with an ``"message"`` key if the column
        is missing or empty (backward-compatible error schema).
    """
    if col not in data.columns:
        return {"message": f"Column '{col}' not found in the DataStream."}

    series = data[col].dropna()
    if series.empty:
        return {
            "effective_sample_size": None,
            "message": "No data available for computation.",
        }

    n = len(series)
    if n < 3:
        return n

    tau = _estimate_tau_int_from_series(series.values)
    return max(1, int(np.ceil(n / tau)))
