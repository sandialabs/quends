r"""
DataStream Class
================
This tutorial covers single-trace analysis with the :class:`~quends.DataStream`
object on real gyrokinetic turbulence data (one GX run and one CGYRO run).

It demonstrates the core single-trace features:

* **Loading & plotting** a raw time-series,
* **Stationarity testing** (Augmented Dickey-Fuller),
* **Trimming** to the steady-state portion (two equivalent calling styles),
* **Effective Sample Size (ESS)** and autocorrelation-aware **statistics**,
* **saving / re-loading** a trimmed stream and handling
  **non-stationary** inputs.

For analysing **multiple runs together**, see the *Ensemble Analysis* guide; for
noisy signals where stationarity is hard to assess, see the *RobustWorkflow*
guide.

The GX trim/statistics parameters (``method="threshold"``, ``window_size=50``,
``start_time=100``, ``threshold=0.1``, ``method="non-overlapping"``) and the
CGYRO parameters (``method="threshold"``, ``window_size=100``,
``threshold=0.1``) follow the QUENDS analysis notebooks.
"""

# %%
# Import QUENDS
import glob
import os
import tempfile

import quends as qnds
from quends.postprocessing.loader import JsonLoader
from quends.postprocessing.writer import JsonWriter

COL = "HeatFlux_st"  # the heat-flux observable carried by the GX files
plotter = qnds.Plotter()

# %%
# GX Data Analysis
# ----------------
# The GX data ships in ``data/gx`` and the CGYRO data in ``data/cgyro``.

# %%
# Single Trace
# ~~~~~~~~~~~~
# **Input case: a single simulation output.** We analyse one GX run.

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
ss_start = trimmed.trim_metadata.get("sss_start")
print("ds.trim            -> sss_start:", ss_start)
trimmed.head()

# %%
# Plot of the trace with the detected steady-state start
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Plot using **the exact steady-state start that the trim found**
# (``trimmed.trim_metadata["sss_start"]``) via ``steady_state_plot`` -- so the
# annotation matches the trim instead of re-detecting with other parameters.
# The post-steady-state mean is drawn over the steady region. (For a
# ``std`` / ``QuantileTrimStrategy`` trim you can pass ``show_std_bands=True`` to
# add the ±1/2/3 std bands; threshold-based trims are shown without them.)
plot = plotter.steady_state_plot(ds, [COL], steady_state_start=ss_start, show=True)

# %%
# Equivalently, ``steady_state_automatic_plot`` re-detects the start, but it must
# be given **the same trim parameters** to match:
plot = plotter.steady_state_automatic_plot(
    ds, [COL], method="threshold", threshold=0.1, batch_size=50, start_time=100,
    show=True,
)

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
# Save the trimmed stream
# ^^^^^^^^^^^^^^^^^^^^^^^
# The postprocessing layer can serialise a ``DataStream`` (data + operation
# history) to JSON with :class:`~quends.postprocessing.writer.JsonWriter`, and
# read it back with :class:`~quends.postprocessing.loader.JsonLoader`.
trimmed_path = os.path.join(tempfile.mkdtemp(), "trimmed_gx.json")
JsonWriter(trimmed_path).save(trimmed)

# %%
# Other input scenarios
# ----------------------

# %%
# Already-trimmed data
# ~~~~~~~~~~~~~~~~~~~~~
# Load the trimmed stream we just saved and run the pipeline on it. Because the
# data is already in steady state, this doubles as a check that trimming behaves
# as expected.
reloaded = JsonLoader(trimmed_path).load()
print("reloaded rows:", len(reloaded), "| variables:", list(reloaded.data.columns))

# %%
# Plot the (already-trimmed) raw trace.
plot = plotter.trace_plot(reloaded, [COL], show=True)

# %%
# It now tests as stationary -- there is no transient left to remove.
print("reloaded is_stationary:", reloaded.is_stationary(COL))

# %%
# Trimming again keeps essentially the same steady-state trace, and the
# statistics reproduce the original trim -- confirming the pipeline is
# consistent.
re_trimmed = reloaded.trim(method="threshold", threshold=0.1, window_size=50, start_time=100)
print("rows: original trim =", len(trimmed), "| reloaded =", len(reloaded),
      "| re-trimmed =", len(re_trimmed))
plot = plotter.trace_plot(re_trimmed, [COL], show=True)

# %%
print("reloaded ESS  :", reloaded.effective_sample_size())
print("reloaded stats:", reloaded.compute_statistics(method="non-overlapping"))

# %%
# Non-stationary / failed steady-state detection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Not every channel reaches steady state. Several GX observables in
# ``tprim_2_4`` are non-stationary -- here ``Phi2_t`` (the electrostatic
# potential energy), which keeps drifting. (``Wg_st`` was suggested but tests as
# stationary in this run, so we use a genuinely non-stationary column.) The
# Augmented Dickey-Fuller test flags it, and a plain trim cannot find a clean
# steady state. For signals like this, use the *RobustWorkflow* guide.
ds_ns = qnds.from_csv("data/gx/tprim_2_4.out.csv", "Phi2_t")
print("Phi2_t is_stationary:", ds_ns.is_stationary("Phi2_t"))
plot = plotter.trace_plot(ds_ns, ["Phi2_t"], show=True)
ns_trim = ds_ns.trim(method="threshold", threshold=0.1, window_size=50, start_time=100)
print("Phi2_t trim_metadata:", ns_trim.trim_metadata)

# %%
# UQ Analysis
# -----------
# Convenience accessors return just the piece you need from the trimmed data.

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
# ``Q_D/Q_GBD`` and the trim parameters follow the CGYRO notebook
# (``method="threshold"``, ``window_size=100``, ``threshold=0.1``).
CG = "Q_D/Q_GBD"
cg = qnds.from_csv("data/cgyro/output_nu0_50.csv", CG)
print("cgyro rows:", len(cg))
cg.head()

# %%
# Raw trace and stationarity check.
plot = plotter.trace_plot(cg, [CG], show=True)
print("cgyro is_stationary (raw):", cg.is_stationary(CG))

# %%
# Trim to the steady-state portion and plot it at the detected start.
cg_trimmed = cg.trim(method="threshold", window_size=100, start_time=0.0, threshold=0.1)
cg_ss = cg_trimmed.trim_metadata.get("sss_start")
print("cgyro sss_start:", cg_ss)
plot = plotter.steady_state_plot(cg, [CG], steady_state_start=cg_ss, show=True)

# %%
# Effective sample size and statistics for the CGYRO run.
print("cgyro ESS:", cg_trimmed.effective_sample_size())
cg_stats = cg_trimmed.compute_statistics(method="non-overlapping")
print(cg_stats)
qnds.Exporter().display_dataframe(cg_stats)
