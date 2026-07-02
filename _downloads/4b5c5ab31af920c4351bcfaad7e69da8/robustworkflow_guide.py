r"""
RobustWorkflow
==============
This tutorial follows the ``robust_workflow`` notebook and demonstrates the
:class:`~quends.RobustWorkflow` class. The workflow can process data streams
that are **not** cleanly stationary: with ``operate_safe=False`` it will, when
needed, drop an initial fraction of the stream to find a stationary tail and
return "ball-park" statistics (flagged in the result ``metadata``) rather than
aborting.

For each signal the procedure is the same as in the notebook:

#. build / load the signal into a :class:`~quends.DataStream`,
#. set up ``RobustWorkflow(operate_safe=False, verbosity=2)`` (high verbosity so
   the intermediate steps are plotted),
#. plot the raw signal with
   :meth:`~quends.RobustWorkflow.plot_signal_basic_stats`,
#. call ``process_data_stream`` to get the statistics, and
#. re-plot the signal with its mean, confidence interval, and SSS start.

For clean single traces the *DataStream Class* guide is simpler; for many runs
see *Ensemble Analysis*. ``RobustWorkflow`` assumes equally spaced time points.
"""

# %%
# Setup
# -----
# ``RobustWorkflow`` *displays* its plots (it calls ``plt.show()`` and then
# closes the figure). The small ``keep_figures`` helper keeps those figures open
# just long enough for the documentation gallery to capture the workflow's own
# output -- it is not needed in a notebook or script.
import contextlib
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import quends as qnds


@contextlib.contextmanager
def keep_figures():
    _close = plt.close
    plt.close = lambda *a, **k: None
    try:
        yield
    finally:
        plt.close = _close


# %%
# Synthetic data: linear transient to a plateau with some noise
# -------------------------------------------------------------
# A linear ramp into a flat signal, plus noise. With ``verbosity=2`` the
# workflow shows its intermediate steps: the raw signal, the autocorrelation
# used to set the averaging window, and the smoothed signal with the detected
# start of statistical steady state (SSS).
arr_time = np.linspace(start=0, stop=1000, num=1001)
arr_signal = np.zeros_like(arr_time)
n_pts = arr_signal.shape[0]
for i_arr in range(100):
    arr_signal[i_arr] = 0.05 * arr_time[i_arr]
for i_arr in range(100, n_pts):
    arr_signal[i_arr] = arr_signal[99]
np.random.seed(42)
arr_signal += np.random.normal(loc=0.0, scale=0.1, size=arr_signal.shape)

my_label = "Linear Transient with Noise"
ds_flat = qnds.DataStream(pd.DataFrame({"time": arr_time, "flux": arr_signal}))
col = "flux"

my_wrkflw = qnds.RobustWorkflow(operate_safe=False, verbosity=2)
with keep_figures():
    # Plot raw signal
    my_wrkflw.plot_signal_basic_stats(ds_flat, col, label=my_label)
    # Get statistics
    my_stats = my_wrkflw.process_data_stream(ds_flat, col)
    # Plot trace with mean and start of steady state
    if not my_stats[col]["metadata"]["mitigation"] == "Drop":
        my_wrkflw.plot_signal_basic_stats(ds_flat, col, stats=my_stats, label=my_label)

# %%
# The dictionary ``my_stats`` holds the numerical results; ``metadata`` reveals
# how the data was processed. Here ``mitigation: None`` / ``status: Regular``
# means no special handling was needed.
pprint.pprint(my_stats)

# %%
# Regular signals: CGYRO
# ----------------------
# A well-behaved CGYRO run. The observable is the third column (after the index
# and ``time``), exactly as in the notebook.
data_paths = ["data/cgyro/output_nu0_02.csv"]
col = pd.read_csv(data_paths[0]).columns[2]  # 3rd column (after index, time)
data_stream = qnds.from_csv(data_paths[0], col)
print("The data stream contains the following variables:")
for column, name in enumerate(data_stream.variables()):
    print(f"{column}: {name}")

my_wrkflw = qnds.RobustWorkflow(operate_safe=False, verbosity=2)
with keep_figures():
    my_wrkflw.plot_signal_basic_stats(data_stream, col, label=data_paths[0])
    my_stats = my_wrkflw.process_data_stream(data_stream, col)
    if not my_stats[col]["metadata"]["mitigation"] == "Drop":
        my_wrkflw.plot_signal_basic_stats(
            data_stream, col, stats=my_stats, label=data_paths[0]
        )

# %%
# Most of this signal is in SSS: the mean is steady and its high standard
# deviation gives the steady-state criterion plenty of wiggle room.
pprint.pprint(my_stats)

# %%
# Regular signals: GX
# -------------------
# A GX heat-flux trace (``HeatFlux_st``).
data_paths = ["data/gx/tprim_2_4.out.csv"]
data_stream = qnds.from_csv(data_paths[0], "HeatFlux_st")
print("The data stream contains the following variables:")
for column, name in enumerate(data_stream.variables()):
    print(f"{column}: {name}")
col = "HeatFlux_st"

my_wrkflw = qnds.RobustWorkflow(operate_safe=False, verbosity=2)
with keep_figures():
    my_wrkflw.plot_signal_basic_stats(data_stream, col, label=data_paths[0])
    my_stats = my_wrkflw.process_data_stream(data_stream, col)
    if not my_stats[col]["metadata"]["mitigation"] == "Drop":
        my_wrkflw.plot_signal_basic_stats(
            data_stream, col, stats=my_stats, label=data_paths[0]
        )

# %%
# Here the decorrelation length is short, so the smoothed signal is more
# variable and about half the trace is identified as steady state.
pprint.pprint(my_stats)
