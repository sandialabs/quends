import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def power_law_model(n, A, p):
    return A / (n**p)


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
    for lag in lag_set:
        L = int(min(lag, n_blocks - 1))
        if L < 1:
            continue
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


def _compute_ess(data, col, alpha):
    if col not in data.columns:
        return {"message": f"Column '{col}' not found in the DataStream."}

    series = data[col].dropna()
    if series.empty:
        return {
            "effective_sample_size": None,
            "message": "No data available for computation.",
        }

    n = len(series)
    acf_values = acf(series, nlags=n // 4)
    threshold = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    significant_acf = acf_values[1:][np.abs(acf_values[1:]) > threshold]
    ess = n / (1 + 2 * np.sum(np.abs(significant_acf)))

    return int(np.ceil(ess))
