r"""
RobustWorkflow: Difficult Cases & Batch Processing
==================================================
This tutorial continues the *RobustWorkflow* guide with the harder cases from
the ``robust_workflow`` notebook:

* a **non-stationary signal**, where the workflow drops successive initial
  fractions until the tail is stationary,
* the same signal with a stricter ``n_pts_min``, where the workflow **gives up**
  and returns "ball-park" (``AdHoc``) statistics,
* a **stationary signal with no steady state** found because of deliberately bad
  hyperparameters, and
* **batch processing** of several runs followed by a flux-vs-collisionality
  plot.

Throughout, ``operate_safe=False`` lets the workflow return a result (flagged in
``metadata``) instead of aborting. See the *RobustWorkflow* guide for the basic
single-signal procedure.
"""

# %%
# Setup
# -----
# As in the basic guide, ``keep_figures`` keeps the workflow's displayed figures
# open long enough for the documentation gallery to capture them.
import contextlib
import re

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
# A signal that is not stationary
# -------------------------------
# This signal was built by adding a linear trend to a stationary signal. When
# the full stream is analysed it is found non-stationary; rather than abort, the
# workflow repeatedly drops a fraction of the data (25% by default) to see
# whether the tail becomes stationary, and here it eventually succeeds.
data_paths = ["data/testdata/non-stat.csv"]
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
print("metadata:", my_stats[col]["metadata"])

# %%
# ... and when it should give up
# ------------------------------
# With a stricter ``n_pts_min`` the workflow stops dropping data once the
# remaining stream gets too short. It then declares the stream non-stationary
# and -- because ``operate_safe=False`` -- returns ad-hoc statistics
# (``status: NoStatSteadyState`` / ``mitigation: AdHoc``) based on the tail of
# the signal, rather than failing.
my_wrkflw = qnds.RobustWorkflow(operate_safe=False, verbosity=2, n_pts_min=1000)
with keep_figures():
    my_wrkflw.plot_signal_basic_stats(data_stream, col, label=data_paths[0])
    my_stats = my_wrkflw.process_data_stream(data_stream, col)
    if not my_stats[col]["metadata"]["mitigation"] == "Drop":
        my_wrkflw.plot_signal_basic_stats(
            data_stream, col, stats=my_stats, label=data_paths[0]
        )

# %%
print("metadata:", my_stats[col]["metadata"])

# %%
# Stationary, but no steady state found
# -------------------------------------
# Here the data is stationary, but deliberately bad hyperparameters
# (``max_lag_frac=0.05``, ``decor_multiplier=1.0``, ``std_dev_frac=0.001``,
# ``fudge_fac=0.0``) give a short averaging window and a tiny deviation
# tolerance, so no SSS segment can be found. An ad-hoc result (based on the last
# third of the signal) is returned instead.
data_path = "data/cgyro/output_nu0_02.csv"
ds0 = qnds.from_csv(data_path, "Q_D/Q_GBD")
col = "Q_D/Q_GBD"
my_wrkflw0 = qnds.RobustWorkflow(
    operate_safe=False,
    verbosity=2,
    max_lag_frac=0.05,
    decor_multiplier=1.0,
    std_dev_frac=0.001,
    fudge_fac=0.0,
)
with keep_figures():
    my_stats0 = my_wrkflw0.process_data_stream(ds0, col)
    if not my_stats0[col]["metadata"]["mitigation"] == "Drop":
        my_wrkflw0.plot_signal_basic_stats(ds0, col, my_stats0, label=data_path)

# %%
print("metadata:", my_stats0[col]["metadata"])

# %%
# Batch processing
# ----------------
# Process a set of CGYRO runs at different collisionalities, gather the mean and
# its uncertainty for each, and plot the flux versus collisionality.
data_paths = [
    "data/cgyro/output_nu0_02.csv",
    "data/cgyro/output_nu0_05.csv",
    "data/cgyro/output_nu0_10.csv",
    "data/cgyro/output_nu0_50.csv",
    "data/cgyro/output_nu1_0.csv",
]
col = pd.read_csv(data_paths[0]).columns[2]

my_wrkflw = qnds.RobustWorkflow(operate_safe=False, verbosity=0)
flux_means = np.empty((len(data_paths),), dtype=float)
flux_unc = np.empty((len(data_paths),), dtype=float)

with keep_figures():
    for i_data, data_path in enumerate(data_paths):
        print(f"\nProcessing {data_path}:")
        data_stream = qnds.from_csv(data_path, col)
        my_stats = my_wrkflw.process_data_stream(data_stream, col)
        if not my_stats[col]["metadata"]["mitigation"] == "Drop":
            my_wrkflw.plot_signal_basic_stats(
                data_stream, col, stats=my_stats, label=data_path
            )
        flux_means[i_data] = my_stats[col]["mean"]
        flux_unc[i_data] = my_stats[col]["mean_uncertainty"]

# %%
# Flux mean (with uncertainty) versus collisionality :math:`\nu`.
nu_values = []
for path in data_paths:
    match = re.search(r"nu([0-9_]+)\.csv", path)
    if match:
        nu_values.append(float(match.group(1).replace("_", ".")))

fig, ax = plt.subplots(figsize=(8, 6))
ax.errorbar(
    nu_values, flux_means, yerr=flux_unc, fmt="o", capsize=5,
    label="Flux Mean with Uncertainty",
)
ax.set_xlabel(r"$\nu$", size=16)
ax.set_ylabel(col, size=16)
ax.legend()
ax.grid(True, alpha=0.3)
