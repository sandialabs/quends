# Getting started

QUENDS (Quantifying Uncertainty in ENsemble Data Streams) turns raw
time-series simulation output into trustworthy statistics: it trims transients,
detects steady state, and quantifies uncertainty for single signals and for
ensembles of runs.

## Installation

```bash
pip install quends
```

To work from a checkout (editable install with the development extras):

```bash
git clone https://github.com/sandialabs/quends.git
cd quends
pip install -e ".[dev]"
```

## Basic usage

The core object is the **`DataStream`**. A typical single-signal analysis is
three steps — **load → trim → quantify**:

```python
import quends as qnds

# 1. Load one signal (plus its time column) from a CSV file.
ds = qnds.from_csv("data/cgyro/output_nu0_50.csv", "Q_D/Q_GBD")

# 2. Trim the warm-up transient, keeping only the steady-state region.
trimmed = ds.trim(method="threshold", window_size=100, threshold=0.1)

# 3. Compute statistics with an honest uncertainty estimate.
stats = trimmed.compute_statistics()
print(stats)
```

```text
{'Q_D/Q_GBD': {'mean': 25.07, 'mean_uncertainty': 1.14, 'window_size': 85,
               'effective_sample_size': 42, ...}}
```

Every loader takes the file and **one** variable name and returns a
`DataStream` that holds just `time` and that signal. Use `ds.variables()` to
list available columns and `ds.data` to reach the underlying pandas
`DataFrame`.

## Ensembles in one step

For a collection of runs, build an `Ensemble` and compute an ensemble estimate
(`ensemble_average`, `pooled_block_means`, or `ivw_member_means`):

```python
paths = ["run01.csv", "run02.csv", "run03.csv"]
ens = qnds.Ensemble.from_files(paths, "Q_D/Q_GBD")
result = ens.compute_uncertainty(method="ensemble_average")
print(result)
```

## Where to next

- **{doc}`User Guide <user_guide/index>`** — the concepts and the recommended
  workflow for each part of the library (loading, trimming, statistics,
  ensembles, plotting, workflows).
- **{doc}`Gallery of examples <auto_tutorials/index>`** — runnable, end-to-end
  tutorials.
- **{doc}`API Reference <autoapi/index>`** — the full, auto-generated API.
