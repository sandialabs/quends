import numpy as np
import pandas as pd

from ..base.data_stream import DataStream
from ..base.history import DataStreamHistory, DataStreamHistoryEntry


def from_numpy(np_array, variable, time=None):
    """
    Load a single-variable data stream (with a time column) from a NumPy array.

    To match the other loaders, the returned :class:`DataStream` carries a
    ``time`` column alongside the named ``variable``. Because a NumPy array has
    no intrinsic time axis, the time values come from one of three sources:

    * **1D array + ``time``** — the supplied ``time`` array (same length).
    * **1D array, no ``time``** — a synthesized integer index ``0, 1, 2, ...``.
    * **Nx2 array** — interpreted as two columns; the (strictly increasing)
      column is taken as ``time`` and the other as ``variable``. If neither is
      monotonic, the first column is assumed to be time.

    Args:
        np_array (np.ndarray): A 1D array of values, or an Nx2 array
            ``[time, variable]``.
        variable (str): The column name to assign to the data values.
        time (array-like, optional): Explicit time values for the 1D case
            (must match the length of ``np_array``). Ignored for Nx2 input.

    Returns:
        DataStream: A DataStream containing ``[time, variable]``.

    Raises:
        ValueError: If the input is not a NumPy array of a supported shape, or
            if a supplied ``time`` array has the wrong length.
    """
    if not isinstance(np_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    arr = np_array
    # Treat an Nx1 column vector as 1D.
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()

    if arr.ndim == 1:
        values = np.asarray(arr, dtype=float)
        if time is not None:
            t = np.asarray(time, dtype=float).ravel()
            if t.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Length of 'time' ({t.shape[0]}) does not match the data "
                    f"length ({values.shape[0]})."
                )
            time_resolution = "explicit"
        else:
            # No intrinsic time axis -> synthesize a uniform integer index.
            t = np.arange(values.shape[0], dtype=float)
            time_resolution = "synthesized"
        df = pd.DataFrame({"time": t, variable: values})

    elif arr.ndim == 2 and arr.shape[1] == 2:
        c0 = np.asarray(arr[:, 0], dtype=float)
        c1 = np.asarray(arr[:, 1], dtype=float)

        def _strictly_increasing(x):
            return x.shape[0] > 1 and bool(np.all(np.diff(x) > 0))

        if _strictly_increasing(c0):
            t, v = c0, c1
        elif _strictly_increasing(c1):
            t, v = c1, c0
        else:
            # Neither column is monotonic; assume column 0 is time by convention.
            t, v = c0, c1
        df = pd.DataFrame({"time": t, variable: v})
        time_resolution = "from_array"

    else:
        raise ValueError(
            "from_numpy expects a 1D array (optionally with time=...) or an Nx2 "
            f"array [time, variable]; got shape {np_array.shape}."
        )

    history = DataStreamHistory()
    history.append(
        DataStreamHistoryEntry(
            operation_name="load",
            parameters={
                "loader": "from_numpy",
                "source": "<ndarray>",
                "variable": variable,
                "time_column": "time",
                "time_resolution": time_resolution,
            },
        )
    )
    return DataStream(df, history=history)
