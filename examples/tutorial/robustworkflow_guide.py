r"""
RobustWorkflow
==============
This tutorial follows the ``robust_workflow`` notebook and demonstrates the
:class:`~quends.RobustWorkflow` class. The workflow can process data streams
that are **not** cleanly stationary: with ``operate_safe=False`` it repeatedly
drops an initial fraction of the stream to find a stationary tail, and returns
"ball-park" statistics (clearly flagged in the result ``metadata``) rather than
aborting when no statistical steady state (SSS) can be found.

For each signal the procedure is the same as in the notebook:

#. load the signal into a :class:`~quends.DataStream`,
#. set up ``RobustWorkflow(operate_safe=False, verbosity=...)``,
#. call ``process_data_stream`` to get the statistics, and
#. plot the signal with its mean, confidence interval, and SSS start via
   :meth:`~quends.RobustWorkflow.plot_signal_basic_stats`.

For clean single traces the *DataStream Class* guide is simpler; for many runs
see *Ensemble Analysis*. ``RobustWorkflow`` assumes equally spaced time points.
"""

# %%
# Setup
# -----
# ``RobustWorkflow`` *displays* its plots (it calls ``plt.show()`` and then
# closes the figure). The small ``keep_figures`` helper below keeps those
# figures open just long enough for the documentation gallery to capture the
# workflow's own output -- it is not needed in a notebook or script.
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
# A signal with a transient
# --------------------------
# A synthetic signal: a linear ramp into a flat plateau, plus noise. With
# ``verbosity=2`` the workflow shows its intermediate steps -- the raw signal,
# the autocorrelation used to set the averaging window, and the smoothed signal
# with the detected SSS start.
arr_time = np.linspace(0, 1000, 1001)
arr_signal = np.zeros_like(arr_time)
arr_signal[:100] = 0.05 * arr_time[:100]
arr_signal[100:] = arr_signal[99]
np.random.seed(42)
arr_signal += np.random.normal(0.0, 0.1, size=arr_signal.shape)
ds_transient = qnds.DataStream(pd.DataFrame({"time": arr_time, "flux": arr_signal}))

rw = qnds.RobustWorkflow(operate_safe=False, verbosity=2)
with keep_figures():
    stats = rw.process_data_stream(ds_transient, "flux")

# %%
# The result dictionary holds the numerical statistics; ``metadata`` reports how
# the data was processed. Here ``status: Regular`` / ``mitigation: None`` means
# no special handling was needed.
pprint.pprint(stats)

# %%
# Regular signals: GX and CGYRO
# -----------------------------
# Well-behaved runs need no mitigation. We process each, then plot the signal
# with the mean, confidence interval, and SSS start overlaid (passing the
# ``stats`` back to ``plot_signal_basic_stats``).
rw = qnds.RobustWorkflow(operate_safe=False, verbosity=0)

ds_gx = qnds.from_csv("data/gx/ens_run_0006.csv", "HeatFlux_st")
gx_stats = rw.process_data_stream(ds_gx, "HeatFlux_st")
print("GX status:", gx_stats["HeatFlux_st"]["metadata"])
with keep_figures():
    rw.plot_signal_basic_stats(
        ds_gx, "HeatFlux_st", stats=gx_stats, label="GX run (HeatFlux_st)"
    )

# %%
ds_cg = qnds.from_csv("data/cgyro/output_nu0_50.csv", "Q_D/Q_GBD")
cg_stats = rw.process_data_stream(ds_cg, "Q_D/Q_GBD")
print("CGYRO status:", cg_stats["Q_D/Q_GBD"]["metadata"])
with keep_figures():
    rw.plot_signal_basic_stats(
        ds_cg, "Q_D/Q_GBD", stats=cg_stats, label="CGYRO run (Q_D/Q_GBD)"
    )

# %%
# A non-stationary signal
# -----------------------
# This signal adds a linear trend to stationary noise, so the full stream is not
# stationary. With ``operate_safe=False`` the workflow drops successive initial
# fractions until the tail is stationary, and records what it did in
# ``metadata`` (note the non-``Regular`` status / mitigation).
np.random.seed(0)
t = np.linspace(0, 1000, 1001)
trend = 0.01 * t
sig = trend + np.random.normal(0.0, 1.0, size=t.shape)
ds_nonstat = qnds.DataStream(pd.DataFrame({"time": t, "signal": sig}))

rw = qnds.RobustWorkflow(operate_safe=False, verbosity=1)
ns_stats = rw.process_data_stream(ds_nonstat, "signal")
print("non-stationary metadata:", ns_stats["signal"]["metadata"])
with keep_figures():
    rw.plot_signal_basic_stats(
        ds_nonstat, "signal", stats=ns_stats, label="Non-stationary signal"
    )
