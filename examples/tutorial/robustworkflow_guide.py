r"""
RobustWorkflow
==============
:class:`~quends.RobustWorkflow` analyses noisy data streams where stationarity
or the start of statistical steady state (SSS) is hard to assess. It augments
the base :class:`~quends.DataStream` statistics with:

* a **stationarity assessment** that progressively shortens the stream to test
  whether its tail is stationary,
* a **robust SSS detection** based on the smoothed mean, and
* **"ball-park" statistics** when no clean steady state is found (``operate_safe
  = False``) instead of failing outright.

Use it as a more forgiving alternative to a direct ``DataStream.trim`` when a
signal is noisy or its transient is poorly behaved. (For clean single traces,
the *DataStream Class* guide is simpler; for many runs, see *Ensemble
Analysis*.) ``RobustWorkflow`` assumes equally spaced time points.
"""

# %%
# Import QUENDS
import glob

import matplotlib.pyplot as plt

import quends as qnds

COL = "HeatFlux_st"

# %%
# Data Loading
# ------------
# Load a single GX run as a :class:`~quends.DataStream`.
gx_files = sorted(glob.glob("data/gx/ens_run_*.csv"))
ds = qnds.from_csv(gx_files[0], COL)
print("loaded:", gx_files[0], "| rows:", len(ds))

# %%
# Running the robust workflow
# ---------------------------
# Construct the workflow and process the stream. ``operate_safe=True`` insists
# on stationarity and a clear SSS segment; ``start_time`` discards an initial
# window before the analysis begins.
rw = qnds.RobustWorkflow(operate_safe=True, verbosity=0)
results = rw.process_data_stream(ds, COL, start_time=100.0)

stats = results[COL]
print("status            :", stats["metadata"]["status"], "/", stats["metadata"]["mitigation"])
print("SSS start         :", round(float(stats["sss_start"]), 2))
print("mean              :", round(stats["mean"], 4))
print("mean uncertainty  :", round(stats["mean_uncertainty"], 4))
print("confidence interval:", tuple(round(x, 4) for x in stats["confidence_interval"]))
print("effective samples :", stats["effective_sample_size"])

# %%
# Visualising the result
# ----------------------
# ``RobustWorkflow`` ships :meth:`~quends.RobustWorkflow.plot_signal_basic_stats`
# for a quick look; here we draw the signal with the robust mean, its confidence
# band, and the detected SSS start so the figure is captured in the gallery.
df = ds.data
ci_lo, ci_hi = stats["confidence_interval"]
sss = float(stats["sss_start"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["time"], df[COL], lw=1.0, alpha=0.7, label=COL)
ax.axvline(sss, color="tab:red", ls="--", label=f"SSS start = {sss:.1f}")
ax.axhline(stats["mean"], color="black", lw=2.0, label=f"mean = {stats['mean']:.3f}")
ax.axhspan(ci_lo, ci_hi, color="tab:green", alpha=0.2, label="95% CI")
ax.set_xlabel("time")
ax.set_ylabel(COL)
ax.set_title("RobustWorkflow: robust mean, CI, and SSS start")
ax.legend(loc="lower right", fontsize="small")
ax.grid(True, alpha=0.3)

# %%
# "Ball-park" statistics for hard cases
# -------------------------------------
# With ``operate_safe=False`` the workflow still returns an estimate even when
# the stream is not cleanly stationary -- including the full raw trace with its
# transient. The ``metadata`` records any mitigation that was applied, so you
# know how much to trust the number.
rw_unsafe = qnds.RobustWorkflow(operate_safe=False, verbosity=0)
ballpark = rw_unsafe.process_data_stream(ds, COL, start_time=0.0)[COL]
print("ball-park status :", ballpark["metadata"]["status"], "/", ballpark["metadata"]["mitigation"])
print("ball-park mean   :", round(ballpark["mean"], 4))
print("ball-park CI     :", tuple(round(x, 4) for x in ballpark["confidence_interval"]))
