"""Shared helpers for the single-variable preprocessing loaders.

The single-variable API loads one named ``variable`` plus a ``time`` column.
This module centralizes the logic that decides which column of an input table
is the time axis, so every loader resolves it identically and standardizes the
output column name to ``"time"``.
"""

import numpy as np
import pandas as pd

# Case-insensitive names commonly used for a time axis.
TIME_ALIASES = {
    "time", "t", "times", "time_s", "timestamp",
    "tnorm", "t_norm", "time_norm",
}


def resolve_time_column(df, variable):
    """Identify the time column of ``df`` (excluding ``variable``).

    Resolution cascade:
      1. **Name match** — a column whose (stripped, lower-cased) name is a known
         time alias (``time``, ``t``, ...).
      2. **Content / monotonicity** — among the remaining columns, a numeric
         column that is monotonically non-decreasing (``is_monotonic_increasing``;
         equal consecutive values are allowed) with more than one unique value.
         With exactly two columns this naturally resolves the "other" column to
         time (elimination). If several columns qualify, the most evenly spaced
         one (smallest coefficient of variation of its successive differences)
         is chosen.

    Note
    ----
    This dataframe path accepts non-decreasing time. ``from_numpy`` uses a
    stricter *strictly increasing* test for its Nx2 input because, lacking
    column names, it must disambiguate the two columns purely by content.

    Parameters
    ----------
    df : pandas.DataFrame
        The loaded table.
    variable : str
        The data variable the user requested; never treated as the time column.

    Returns
    -------
    str or None
        The name of the resolved time column, or ``None`` if no time-like
        column can be identified (e.g. the requested ``variable`` is itself a
        time series, or the table holds no monotonic axis).
    """
    # If the requested variable is itself a time-like series, there is no
    # separate time column to attach.
    if str(variable).strip().lower() in TIME_ALIASES:
        return None

    others = [c for c in df.columns if c != variable]
    if not others:
        return None

    # 1. Name alias (case-insensitive).
    for c in others:
        if str(c).strip().lower() in TIME_ALIASES:
            return c

    # 2. Content heuristic: strictly increasing numeric column(s).
    candidates = []
    for c in others:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().all() and s.is_monotonic_increasing and s.nunique() > 1:
            candidates.append(c)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Prefer the most evenly spaced column (time is near-uniformly sampled).
        best, best_cv = None, np.inf
        for c in candidates:
            d = np.diff(pd.to_numeric(df[c]).to_numpy(dtype=float))
            mean_d = np.mean(d)
            cv = (np.std(d) / abs(mean_d)) if mean_d != 0 else np.inf
            if cv < best_cv:
                best, best_cv = c, cv
        return best

    return None


def _time_resolution_method(df, variable, time_col):
    """Describe how the time column was identified (for provenance)."""
    if time_col is None:
        return "none"
    if str(time_col).strip().lower() in TIME_ALIASES:
        return "name_alias"
    return "monotonic"


def build_single_variable_frame(df, variable):
    """Return a DataFrame with ``[time, variable]`` (or ``[variable]``).

    The resolved time column is renamed to the canonical ``"time"`` so that the
    rest of the package (trim / steady-state / ensemble averaging) works
    unchanged.
    """
    time_col = resolve_time_column(df, variable)
    if time_col is not None:
        out = df[[time_col, variable]].copy()
        if time_col != "time":
            out = out.rename(columns={time_col: "time"})
        return out
    return df[[variable]].copy()


def load_single_variable(df, variable, *, source, loader):
    """Build a provenance-tagged ``DataStream`` of ``[time, variable]``.

    Records a ``"load"`` history entry (source, variable, loader, resolved time
    column and how it was detected) so the origin of every stream is traceable.
    """
    # Local imports avoid a circular import at module load time.
    from ..base.data_stream import DataStream
    from ..base.history import DataStreamHistory, DataStreamHistoryEntry

    time_col = resolve_time_column(df, variable)
    frame = build_single_variable_frame(df, variable)
    history = DataStreamHistory()
    history.append(
        DataStreamHistoryEntry(
            operation_name="load",
            parameters={
                "loader": loader,
                "source": str(source),
                "variable": variable,
                "time_column": time_col,
                "time_resolution": _time_resolution_method(df, variable, time_col),
            },
        )
    )
    return DataStream(frame, history=history)
