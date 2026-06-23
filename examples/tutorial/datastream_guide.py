r"""
DataStream Class
================
This tutorial walks through a complete QUENDS analysis on real gyrokinetic
turbulence data, covering the inputs QUENDS is designed for:

* **single trace / single simulation output** -- one GX run,
* **ensemble traces / multiple simulation runs** -- a small GX ensemble,
* **already-trimmed data** -- reusing a steady-state slice without re-trimming,
* **non-stationary / failed steady-state detection** -- what the diagnostics
  look like and how trimming falls back.

It demonstrates the core ``DataStream`` and ``Ensemble`` features:

* **Loading & plotting** raw time-series,
* **Stationarity testing** (Augmented Dickey-Fuller),
* **Trimming** to the steady-state portion (two equivalent calling styles),
* **Effective Sample Size (ESS)** and autocorrelation-aware **statistics**,
* **Ensemble uncertainty** via the Ensemble Average, Serialization
  (pooled block means), and Inverse-Variance-Weighted (IVW) approaches.

The trim/statistics parameters used below (``method="threshold"``,
``window_size=50``, ``start_time=100``, ``threshold≈0.1-0.19``,
``method="non-overlapping"``) follow the QUENDS paper analysis notebooks for the
stellarator GX ensemble.
"""

# %%
# Import QUENDS
import glob

import quends as qnds

COL = "HeatFlux_st"  # the heat-flux observable carried by the GX files
plotter = qnds.Plotter()

# %%
# GX Data Analysis
# ----------------
# The GX data ships in ``data/gx`` (eight ensemble members) and the CGYRO data
# in ``data/cgyro``.

# %%
# Single Trace
# ~~~~~~~~~~~~
# **Input case 1: a single simulation output.** We analyse one GX member.

# %%
# Data Loading
# ^^^^^^^^^^^^
gx_files = sorted(glob.glob("data/gx/ens_run_*.csv"))
single_path = gx_files[0]
ds = qnds.from_csv(single_path, COL)
print("loaded:", single_path, "| variables:", ds.variables(), "| rows:", len(ds))
ds.head()

# %%
# Plotting the raw trace
# ^^^^^^^^^^^^^^^^^^^^^^
# The raw trace shows the initial transient followed by a noisy steady state.
plot = plotter.trace_plot(ds, [COL], show=True)

# %%
# Stationary Check
# ^^^^^^^^^^^^^^^^
# The Augmented Dickey-Fuller test reports whether the (raw) signal already
# looks stationary.
print("is_stationary (raw):", ds.is_stationary(COL))

# %%
# Trimming data to obtain the steady-state portion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# QUENDS offers **two equivalent ways** to trim. First, the explicit
# strategy/operation pattern from :mod:`quends.base.trim` -- useful when you
# want to build a strategy once and reuse it:
from quends.base.trim import TrimDataStreamOperation, build_trim_strategy

strat = build_trim_strategy(
    method="threshold", window_size=50, start_time=100, threshold=0.1
)
trimmed = TrimDataStreamOperation(strategy=strat)(ds, column_name=COL)
print("strategy/operation -> sss_start:", trimmed.trim_metadata.get("sss_start"))

# %%
# Second, the convenience wrapper ``DataStream.trim`` -- the same canonical path
# in one call. Both produce the identical steady-state start:
trimmed = ds.trim(method="threshold", threshold=0.1, window_size=50, start_time=100)
print("ds.trim            -> sss_start:", trimmed.trim_metadata.get("sss_start"))
trimmed.head()

# %%
# Plot of the trace with the detected steady-state start
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The automatic steady-state plot overlays the raw trace with the location
# where the steady state is detected to begin.
plot = plotter.steady_state_automatic_plot(ds, variables_to_plot=[COL], show=True)

# %%
# Effective Sample Size
# ^^^^^^^^^^^^^^^^^^^^^
# Autocorrelation means the trimmed series holds fewer *independent* samples
# than rows; ESS quantifies that.
print("ESS (trimmed):", trimmed.effective_sample_size())

# %%
# Statistical Analysis
# ^^^^^^^^^^^^^^^^^^^^
# ``compute_statistics`` returns the mean, an autocorrelation-corrected
# uncertainty, a confidence interval, and the block/window diagnostics.
stats = trimmed.compute_statistics(method="non-overlapping")
print(stats)
qnds.Exporter().display_dataframe(stats)

# %%
# Ensemble Analysis
# ~~~~~~~~~~~~~~~~~
# **Input case 2: multiple simulation runs.** Instead of one long trace we use
# several shorter runs and combine them.

# %%
# Data Loading
# ^^^^^^^^^^^^
ens = qnds.Ensemble.from_files(gx_files, COL)
print("ensemble members:", len(ens.members()))

# %%
# Ensemble Average Approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Average the members onto a common time grid, then analyse that single
# averaged trace exactly like a single run.

# %%
# *Plotting all members together with the ensemble average.*
plot = plotter.plot_ensemble_with_average(
    ens, variables_to_plot=[COL], condensed_legend=True, show=True
)

# %%
# Build the averaged DataStream.
avg = ens.compute_average_ensemble()
print("averaged trace rows:", len(avg))

# %%
# *Stationary check* on the average.
print("avg is_stationary:", avg.is_stationary(COL))

# %%
# *Trim* the averaged trace to its steady-state portion.
avg_trimmed = avg.trim(
    method="threshold", window_size=50, start_time=100, threshold=0.19
)
print("avg sss_start:", avg_trimmed.trim_metadata.get("sss_start"))

# %%
# *Effective sample size* and *statistical analysis* of the averaged trace.
print("avg ESS:", avg_trimmed.effective_sample_size())
ea = ens.compute_uncertainty(method="ensemble_average", column_name=COL)
print("Ensemble Average ->", ea["results"][COL])

# %%
# Serialization (Pooled Block Means) Approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Rather than averaging the traces, pool the per-member block means. Trim every
# member first, then aggregate across members.
ens_trimmed = ens.trim(
    column_name=COL, method="threshold", window_size=50, start_time=100, threshold=0.19
)

# %%
# *Stationary check* (per member) and *effective sample size* for the pooled
# estimator.
print("members stationary:", ens.is_stationary(COL)["results"])
print("ESS (pooled_block_means):", ens_trimmed.effective_sample_size(COL)["results"])

# %%
# *Statistical analysis* -- the serialization estimate of mean and uncertainty.
ser = ens.compute_uncertainty(method="pooled_block_means", column_name=COL)
print("Serialization ->", ser["results"][COL])

# %%
# Inverse-Variance-Weighted (IVW) Approach
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Combine the per-member means weighting each by its inverse variance, so
# better-resolved members count more. Trimming/stationarity are as above.
print("ESS (ivw):", ens_trimmed.effective_sample_size(COL, technique="ivw")["results"])
ivw = ens.compute_uncertainty(method="ivw", column_name=COL)
print("IVW ->", ivw["results"][COL])

# %%
# The three approaches give consistent means with different uncertainty
# budgets -- a useful cross-check on an ensemble.

# %%
# Other input scenarios
# ----------------------

# %%
# Already-trimmed data
# ~~~~~~~~~~~~~~~~~~~~~
# **Input case 3.** A trimmed result is itself a ``DataStream``: feed it
# straight to the statistics without trimming again (and likewise if you load a
# CSV that was trimmed elsewhere).
print("rows already in steady state:", len(trimmed))
print("mean of already-trimmed data:", trimmed.mean())

# %%
# Non-stationary / failed steady-state detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# **Input case 4.** When a strict threshold never clears, the detector falls
# back to ``start_time`` instead of failing. Inspect ``trim_metadata`` to see
# what happened, and pair it with the stationarity test before trusting a mean.
strict = ds.trim(method="threshold", threshold=1e-4, window_size=50, start_time=100)
print("strict-threshold trim_metadata:", strict.trim_metadata)
print("stationary after fallback:", strict.is_stationary(COL))

# %%
# UQ Analysis
# -----------
# Convenience accessors return just the piece you need from the single-trace
# trimmed data.

# %%
# Other statistical methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
print("mean:", trimmed.mean(method="sliding"))
print("mean uncertainty:", trimmed.mean_uncertainty(method="sliding"))
print("confidence interval:", trimmed.confidence_interval())

# %%
# Cumulative statistics track how the estimate stabilises as more samples are
# included.
cumulative = trimmed.cumulative_statistics()
qnds.Exporter().display_dataframe(cumulative)

# %%
# ``additional_data`` exposes the underlying block diagnostics.
print(trimmed.additional_data(method="sliding"))

# %%
# CGYRO Data Analysis
# -------------------
# The same workflow applies to CGYRO output; here the observable is
# ``Q_D/Q_GBD``.
cg = qnds.from_csv("data/cgyro/output_nu0_50.csv", "Q_D/Q_GBD")
print("cgyro rows:", len(cg), "| stationary:", cg.is_stationary("Q_D/Q_GBD"))
cg.head()

# %%
# Trim with the Quantile (std) strategy and plot the detected steady state.
cg_trimmed = cg.trim(method="std", robust=True)
print("cgyro sss_start:", cg_trimmed.trim_metadata.get("sss_start"))
plot = plotter.trace_plot(cg, ["Q_D/Q_GBD"], show=True)

# %%
plot = plotter.steady_state_automatic_plot(cg, variables_to_plot=["Q_D/Q_GBD"], show=True)
