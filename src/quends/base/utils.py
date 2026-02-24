import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf


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
