"""
ensemble_utils.py
-----------------
Module-level utility functions for ensemble analysis.

These functions operate on plain lists of :class:`~quends.base.data_stream.DataStream`
objects, making them reusable from workflow classes, helper scripts, and the
``Ensemble`` class itself without introducing circular dependencies.

Public API
----------
validate_members        Check that a list of DataStreams is valid.
validate_column         Check that a column exists in every member.
get_common_variables    Column names shared by all members.
resolve_cols            Normalize ``column_name`` to a concrete list of strings.
check_time_steps_uniformity
                        Classify the time-step regularity of each member.
interpolate_to_common_time
                        Interpolate all members onto a common regular grid.
direct_average          Average DataStreams that already share the same grid.
compute_average_ensemble
                        Build one averaged DataStream (auto-interpolates if needed).
trim_members            Trim each member and return only non-empty results.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

from quends.base.data_stream import DataStream


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_members(data_streams: List[DataStream]) -> None:
    """
    Raise an informative exception when ``data_streams`` is not a valid
    non-empty list of :class:`~quends.base.data_stream.DataStream` objects.

    Parameters
    ----------
    data_streams : list of DataStream

    Raises
    ------
    ValueError
        If the list is empty.
    TypeError
        If *data_streams* is not a list, or any element is not a DataStream.
    """
    if not isinstance(data_streams, list):
        raise TypeError(
            f"Expected a list of DataStream objects, got {type(data_streams).__name__!r}."
        )
    if not data_streams:
        raise ValueError("Provide a non-empty list of DataStream objects.")
    for i, ds in enumerate(data_streams):
        if not isinstance(ds, DataStream):
            raise TypeError(
                f"All ensemble members must be DataStream instances; "
                f"member {i} is {type(ds).__name__!r}."
            )


def validate_column(data_streams: List[DataStream], column_name: str) -> None:
    """
    Raise an informative exception when *column_name* is missing from any
    ensemble member.

    Parameters
    ----------
    data_streams : list of DataStream
    column_name : str

    Raises
    ------
    TypeError
        If *column_name* is not a string.
    KeyError
        If *column_name* is absent from any member.
    """
    if not isinstance(column_name, str):
        raise TypeError(
            f"column_name must be a string, got {type(column_name).__name__!r}."
        )
    for i, ds in enumerate(data_streams):
        if column_name not in ds.data.columns:
            raise KeyError(
                f"Column '{column_name}' not found in ensemble member {i}. "
                f"Available columns: {list(ds.data.columns)}."
            )


# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

def get_common_variables(data_streams: List[DataStream]) -> List[str]:
    """
    Return sorted column names shared by every member, excluding ``'time'``.

    Parameters
    ----------
    data_streams : list of DataStream

    Returns
    -------
    list of str
    """
    if not data_streams:
        return []
    sets = [set(ds.data.columns) - {"time"} for ds in data_streams]
    return sorted(list(set.intersection(*sets))) if sets else []


def resolve_cols(
    data_streams: List[DataStream],
    column_name: Optional[Any],
) -> List[str]:
    """
    Normalize *column_name* to a concrete list of strings.

    Parameters
    ----------
    data_streams : list of DataStream
        Used to enumerate common variables when *column_name* is ``None``.
    column_name : str, list of str, or None
        ``None`` → all common variables; str → single-element list;
        list → returned as-is.

    Returns
    -------
    list of str
    """
    if isinstance(column_name, str):
        return [column_name]
    if column_name is None:
        return get_common_variables(data_streams)
    return list(column_name)


# ---------------------------------------------------------------------------
# Time-grid utilities
# ---------------------------------------------------------------------------

def check_time_steps_uniformity(
    data_streams: List[DataStream],
    tol: float = 1e-8,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Inspect the time-step regularity of each ensemble member.

    For each member, computes diffs of the ``'time'`` column and classifies as:

    ``"AllEqual"``
        All steps identical (within *tol*).
    ``"AllEqualButLast"``
        All steps equal except the last one.
    ``"NotUniform"``
        Multiple distinct step sizes.

    Parameters
    ----------
    data_streams : list of DataStream
    tol : float
        Absolute tolerance for step-size comparison.
    verbose : bool
        Print per-member diagnostics.

    Returns
    -------
    dict
        ``{"uniform": bool, "majority_step": float, "members": {…}}``
    """
    member_info: Dict[str, Any] = {}
    all_steps: List[float] = []

    for i, ds in enumerate(data_streams):
        if "time" not in ds.data.columns:
            member_info[f"Member {i}"] = {"status": "no_time_column"}
            continue

        times = ds.data["time"].values
        if len(times) < 2:
            member_info[f"Member {i}"] = {"status": "too_few_points"}
            continue

        steps = np.diff(times)
        rounded = np.round(steps / tol) * tol
        uniq = np.unique(rounded).tolist()

        if len(uniq) == 1:
            status = "AllEqual"
        elif (
            len(uniq) == 2
            and np.allclose(rounded[:-1], rounded[0], atol=tol)
            and not np.isclose(rounded[-1], rounded[0], atol=tol)
        ):
            status = "AllEqualButLast"
        else:
            status = "NotUniform"

        all_steps.extend(uniq)
        member_info[f"Member {i}"] = {
            "status": status,
            "unique_steps": uniq,
            "n_steps": len(steps),
            "t_min": float(times[0]),
            "t_max": float(times[-1]),
        }

        if verbose:
            print(f"  Member {i}: status={status}, unique_steps={uniq}")

    # Majority step across all members
    if all_steps:
        step_arr = np.array(all_steps)
        vals, counts = np.unique(np.round(step_arr, 10), return_counts=True)
        majority_step = float(vals[np.argmax(counts)])
    else:
        majority_step = np.nan

    # Overall uniform: all members AllEqual with the same step
    all_equal = all(
        v.get("status") in ("AllEqual", "AllEqualButLast")
        and len(v.get("unique_steps", [])) == 1
        and np.isclose(v["unique_steps"][0], majority_step, atol=tol)
        for v in member_info.values()
        if "status" in v
    )

    return {
        "uniform": all_equal,
        "majority_step": majority_step,
        "members": member_info,
    }


def interpolate_to_common_time(
    data_streams: List[DataStream],
    method: str = "spline",
    tol: float = 1e-8,
    verbose: bool = False,
) -> Tuple[List[DataStream], Dict[str, Any]]:
    """
    Interpolate all ensemble members onto a common, regular time grid.

    The common grid spans ``[min(t_start), max(t_end)]`` across all members
    using the majority time step.

    Parameters
    ----------
    data_streams : list of DataStream
    method : {"spline", "linear"}
        Interpolation method.
    tol : float
        Tolerance for step-size uniformity check.
    verbose : bool
        Print grid diagnostics.

    Returns
    -------
    (new_data_streams, diagnostics)
        ``new_data_streams`` is a plain :class:`list` of :class:`DataStream`.

    Raises
    ------
    ValueError
        If a valid majority step cannot be determined, or if the common grid
        spans zero range.
    """
    step_info = check_time_steps_uniformity(data_streams, tol=tol, verbose=verbose)
    majority_step = step_info["majority_step"]

    if not np.isfinite(majority_step) or majority_step <= 0:
        raise ValueError(
            "Could not determine a valid majority step size from the ensemble members."
        )

    t_min = min(
        v["t_min"] for v in step_info["members"].values() if "t_min" in v
    )
    t_max = max(
        v["t_max"] for v in step_info["members"].values() if "t_max" in v
    )

    if t_max <= t_min:
        raise ValueError(
            f"Common time range is degenerate: t_min={t_min}, t_max={t_max}."
        )

    n_grid = int(np.ceil((t_max - t_min) / majority_step)) + 1
    common_times = t_min + np.arange(n_grid) * majority_step

    if verbose:
        print(
            f"Common grid: t_min={t_min:.4g}, t_max={t_max:.4g}, "
            f"step={majority_step:.4g}, N={n_grid}"
        )

    new_members: List[DataStream] = []
    interp_meta: List[Dict] = []

    for idx, ds in enumerate(data_streams):
        orig_times = ds.data["time"].values
        new_df: Dict[str, np.ndarray] = {"time": common_times}

        for col in ds.data.columns:
            if col == "time":
                continue
            y = ds.data[col].values
            mask = np.isfinite(orig_times) & np.isfinite(y)
            if np.sum(mask) < 2:
                new_df[col] = np.full(len(common_times), np.nan)
                continue
            try:
                if method == "spline" and np.sum(mask) >= 4:
                    f = CubicSpline(orig_times[mask], y[mask], extrapolate=True)
                else:
                    f = interp1d(
                        orig_times[mask],
                        y[mask],
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False,
                    )
                new_df[col] = f(common_times)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(
                        f"  Interpolation failed for member {idx}, col {col}: {exc}"
                    )
                new_df[col] = np.full(len(common_times), np.nan)

        new_members.append(DataStream(pd.DataFrame(new_df)))
        interp_meta.append(
            {
                "member": idx,
                "step_status": step_info["members"]
                .get(f"Member {idx}", {})
                .get("status"),
                "original_t_range": (
                    float(orig_times[0]) if len(orig_times) else np.nan,
                    float(orig_times[-1]) if len(orig_times) else np.nan,
                ),
            }
        )

    diagnostics: Dict[str, Any] = {
        "majority_step": majority_step,
        "t_min": float(t_min),
        "t_max": float(t_max),
        "n_grid": int(n_grid),
        "method": method,
        "step_info": step_info,
        "member_meta": interp_meta,
    }
    return new_members, diagnostics


# ---------------------------------------------------------------------------
# Averaging helpers
# ---------------------------------------------------------------------------

def direct_average(
    data_streams: List[DataStream],
    cols: Optional[List[str]] = None,
    min_coverage: int = 1,
) -> Tuple[DataStream, Dict]:
    """
    Average a list of DataStreams with compatible time grids by stacking and
    computing the per-time-point mean.

    Parameters
    ----------
    data_streams : list of DataStream
        All members must already share (or be compatible with) the same time
        points.
    cols : list of str or None
        Columns to average.  Defaults to all columns common to every member
        (excluding ``'time'``).
    min_coverage : int
        Minimum number of non-NaN members required at a time point for the
        average to be non-NaN.

    Returns
    -------
    (averaged_DataStream, meta)
    """
    if cols is None:
        cols = get_common_variables(data_streams)

    all_times = np.unique(
        np.concatenate(
            [np.round(ds.data["time"].values, 6) for ds in data_streams]
        )
    )
    all_times.sort()
    avg_df = pd.DataFrame({"time": all_times})

    for col in cols:
        member_series = []
        for ds in data_streams:
            if col not in ds.data.columns:
                continue
            sub = ds.data[["time", col]].dropna().copy()
            sub["time"] = np.round(sub["time"], 6)
            s = pd.Series(np.nan, index=np.arange(len(all_times)), dtype=float)
            t_vals = sub["time"].to_numpy(dtype=float)
            v_vals = sub[col].to_numpy(dtype=float)
            idxs = np.searchsorted(all_times, t_vals)
            valid = (idxs >= 0) & (idxs < len(all_times))
            if np.any(valid):
                s.iloc[idxs[valid]] = v_vals[valid]
            member_series.append(s)

        if not member_series:
            avg_df[col] = np.nan
            continue

        stack = np.vstack([s.values for s in member_series])
        count_vals = np.sum(~np.isnan(stack), axis=0)
        with np.errstate(invalid="ignore"):
            mean_vals = np.nanmean(stack, axis=0)
        mean_vals[count_vals < min_coverage] = np.nan
        avg_df[col] = mean_vals

    avg_df = avg_df.dropna(subset=cols, how="all").reset_index(drop=True)
    return DataStream(avg_df), {"n_members": len(data_streams)}


def compute_average_ensemble(
    data_streams: List[DataStream],
    interp_method: str = "spline",
    tol: float = 1e-8,
    min_coverage: int = 1,
    verbose: bool = False,
) -> DataStream:
    """
    Build a single averaged :class:`~quends.base.data_stream.DataStream` from
    ensemble members.

    If all members share the same time grid (detected via
    :func:`check_time_steps_uniformity`), averages directly.
    If grids differ, interpolates all members to a common grid first.

    Parameters
    ----------
    data_streams : list of DataStream
    interp_method : {"spline", "linear"}
        Interpolation method used when grids differ.
    tol : float
        Tolerance for uniformity check.
    min_coverage : int
        Minimum number of members that must contribute to a time point.
    verbose : bool
        Print diagnostics when interpolation is triggered.

    Returns
    -------
    DataStream
        Single averaged trace.

    Raises
    ------
    ValueError
        If *data_streams* is empty.
    """
    if not data_streams:
        raise ValueError("No data streams provided for ensemble averaging.")

    step_info = check_time_steps_uniformity(data_streams, tol=tol)

    if step_info["uniform"]:
        avg_ds, _ = direct_average(data_streams, min_coverage=min_coverage)
        return avg_ds

    if verbose:
        print("Time grids differ; interpolating to common grid.")
    interp_members, _ = interpolate_to_common_time(
        data_streams, method=interp_method, tol=tol, verbose=verbose
    )
    avg_ds, _ = direct_average(interp_members, min_coverage=min_coverage)
    return avg_ds


# ---------------------------------------------------------------------------
# Trimming helper
# ---------------------------------------------------------------------------

def trim_members(
    data_streams: List[DataStream],
    column_name: str,
    strategy: Any = None,
    method: str = "std",
    window_size: int = 10,
    start_time: float = 0.0,
    threshold: Optional[float] = None,
    robust: bool = True,
) -> List[DataStream]:
    """
    Trim each member in *data_streams* and return only non-empty results.

    Either pass a pre-built *strategy* object (any
    :class:`~quends.base.trim.TrimStrategy` subclass), or specify *method* and
    associated parameters so that a strategy is built internally via
    :func:`~quends.base.trim.build_trim_strategy`.

    Parameters
    ----------
    data_streams : list of DataStream
    column_name : str
        Column whose steady-state start drives the trim.
    strategy : TrimStrategy or None
        Pre-built trim strategy.  If ``None``, *method* and companions are used.
    method : str
        Trim strategy name: ``"std"``, ``"threshold"``, ``"rolling_variance"``,
        ``"self_consistent"``, or ``"iqr"``.
    window_size : int
    start_time : float
    threshold : float or None
    robust : bool

    Returns
    -------
    list of DataStream
        Non-empty trimmed members (members whose trimmed result was empty are
        silently dropped).

    Raises
    ------
    ValueError
        If *method* is unrecognised.
    """
    from quends.base.trim import TrimDataStreamOperation, build_trim_strategy  # noqa: PLC0415

    if strategy is None:
        strategy = build_trim_strategy(
            method=method,
            window_size=window_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )

    op = TrimDataStreamOperation(strategy=strategy)
    trimmed = [op(ds, column_name=column_name) for ds in data_streams]
    return [t for t in trimmed if t is not None and not t.data.empty]
