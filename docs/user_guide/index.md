# User guide

This guide explains the main concepts behind QUENDS and the recommended workflow
for each part of the library. It is organised to mirror the package layout, so
each topic below corresponds to a module in the
{doc}`API Reference <../autoapi/index>`.

For complete executable examples, see the
{doc}`Gallery of examples <../auto_tutorials/index>`.

| Topic | Module | Purpose |
|---|---|---|
| Core concepts | — | Introduces data streams, transients, steady state, ensembles, and uncertainty. |
| [Loading data](#loading-data) | `quends.preprocessing` | Explains how to prepare CSV files, DataFrames, time columns, and signal variables. |
| [Trimming & steady-state detection](#trimming-and-steady-state) | `quends.base.trim` | Explains how QUENDS detects transient behavior and identifies steady-state portions. |
| [Statistics & uncertainty](#statistics-and-uncertainty) | `quends.base.data_stream` | Describes summary statistics, confidence intervals, uncertainty estimates, and effective sample size. |
| Single-trace analysis | `quends.base.data_stream` | Shows the recommended workflow for analyzing one simulation output. |
| [Ensemble analysis](#ensembles) | `quends.base.ensemble` | Explains how to analyze multiple simulation runs, align traces, and compute ensemble summaries. |
| [Plotting & export](#plotting-and-export) | `quends.postprocessing` | Describes how to visualize results and export processed outputs. |
| [Workflows](#workflows) | `quends.workflow` | Explains high-level workflow utilities for repeated QUENDS analyses. |
| Method selection | `quends.base.trim` | Helps users choose appropriate trimming, steady-state, and uncertainty methods. |
| Troubleshooting | — | Covers common issues such as non-stationary data, failed steady-state detection, irregular time steps, and inconsistent ensemble lengths. |

QUENDS analyses follow one pipeline:

```text
raw file/array  ->  DataStream  ->  trim (steady state)  ->  statistics + uncertainty
                                         |
                  many runs  ->  Ensemble  ->  ensemble estimate
```

(loading-data)=
## Loading data

The preprocessing layer converts raw simulation output into a `DataStream`.
Each loader takes a source and **one** variable name, and returns a
`DataStream` holding just the `time` column and that signal:

```python
import quends as qnds

ds = qnds.from_csv("output.csv", "Q_D/Q_GBD")    # CSV
ds = qnds.from_gx("run.nc", "HeatFlux_st")         # GX NetCDF
ds = qnds.from_netcdf("data.nc", "flux")           # generic NetCDF
```

The time column is detected automatically (by name, then by monotonicity).
Inspect what you loaded with `ds.variables()` (returns `['time', '<signal>']`)
and reach the underlying pandas `DataFrame` via `ds.data`. To analyse several
signals from one file, load each into its own `DataStream`.

(trimming-and-steady-state)=
## Trimming & steady state

Most simulation traces begin with a transient that must be removed before
statistics are meaningful. `DataStream.trim` detects the start of the
steady-state region and returns a new, trimmed `DataStream`:

```python
trimmed = ds.trim(
    column_name="Q_D/Q_GBD",
    method="std",          # detection method (see below)
    window_size=10,
    start_time=0.0,        # ignore everything before this time
    robust=True,           # median + MAD instead of mean + std
)
```

Available `method` values:

| Method | Idea |
|---|---|
| `std` | Median/MAD (or mean/std) z-score criterion over the tail. |
| `threshold` | Rolling standard deviation on the normalised signal falls below a threshold. |
| `rolling_variance` | Rolling variance falls below a threshold. |
| `self_consistent` | Self-consistent steady-state segment. |
| `iqr` | Interquartile-range based detection. |
| `mean_variation` | Detects where the running mean stops drifting. |

`start_time` acts as a hard lower bound (everything before it is discarded);
the detector then searches from there. The trimmed result records where it cut
in `trimmed.trim_metadata["sss_start"]`.

(statistics-and-uncertainty)=
## Statistics & uncertainty

Once trimmed, compute statistics with an uncertainty estimate that accounts for
autocorrelation:

```python
stats = trimmed.compute_statistics(method="non-overlapping")
```

The result is a `StatsResult` (a `dict` subclass with a `.metadata` attribute)
keyed by column. Each entry includes the `mean`, the `mean_uncertainty`
(standard error of the mean corrected for the **effective sample size**), the
`confidence_interval`, the chosen block `window_size`, the
`effective_sample_size`, and independence diagnostics (Ljung–Box). Convenience
wrappers `mean`, `mean_uncertainty`, `confidence_interval`, and
`effective_sample_size` return just the piece you need.

(ensembles)=
## Ensembles

An `Ensemble` is a collection of `DataStream` members analysed together. Build
one from files or from existing streams:

```python
ens = qnds.Ensemble.from_files(["run01.csv", "run02.csv"], "Q_D/Q_GBD")
result = ens.compute_uncertainty(method="ensemble_average")
```

Three ensemble estimators are available via `method`:

| Estimator (`method`) | What it does |
|---|---|
| `ensemble_average` (T0) | Average the members onto a common time grid, then quantify the single averaged trace. |
| `pooled_block_means` (T1) | Pool block means across all members. |
| `ivw_member_means` (T2) | Inverse-variance-weighted combination of per-member means. |

Members on different time grids are interpolated to a common grid
automatically; empty members are ignored.

(plotting-and-export)=
## Plotting & export

The postprocessing layer provides plotting and serialisation. The `Plotter`
returns the Matplotlib objects (a list of `(fig, axes)`) instead of forcing a
display, so plots compose cleanly in notebooks and scripts:

```python
plotter = qnds.Plotter()
plotter.trace_plot(ds, ["Q_D/Q_GBD"])
plotter.steady_state_automatic_plot(ds, ["Q_D/Q_GBD"], batch_size=10)
```

`Exporter` displays or converts results (`display_dataframe`, `display_json`),
and `JsonWriter` / `JsonLoader` round-trip a `DataStream` (including its
operation history) to and from JSON.

(workflows)=
## Workflows

The workflow layer packages common multi-step analyses end-to-end:

- **`RobustWorkflow`** — robust stationarity assessment and steady-state
  detection for noisy signals, returning "ball-park" statistics when no clean
  steady state is found.
- **Ensemble workflows** — `EnsembleAverageWorkflow`,
  `EnsembleStatisticsWorkflow`, and `BatchEnsembleWorkflow` orchestrate
  loading, trimming, averaging, and statistics across many runs.

See the {doc}`Gallery of examples <../auto_tutorials/index>` for runnable,
end-to-end demonstrations, and the {doc}`API Reference <../autoapi/index>` for
full signatures.
