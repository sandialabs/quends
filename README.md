# Quantification of Uncertainty in ENsembles of Data Streams (QUENDS)

#### Evans Etrue Howard, Abeyah Calpatura, Pieterjan Robbe, Bert Debusschere

[![Coverage Status](https://coveralls.io/repos/github/sandialabs/quends/badge.svg?branch=main)](https://coveralls.io/github/sandialabs/quends?branch=main)
[![Deploy to GitHub Pages](https://github.com/sandialabs/quends/actions/workflows/deployment.yml/badge.svg)](https://github.com/sandialabs/quends/actions/workflows/deployment.yml)
[![Run Tests](https://github.com/sandialabs/quends/actions/workflows/python-tests.yml/badge.svg)](https://github.com/sandialabs/quends/actions/workflows/python-tests.yml)
[![pages-build-deployment](https://github.com/sandialabs/quends/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/sandialabs/quends/actions/workflows/pages/pages-build-deployment)

## Overview
This project focuses on uncertainty quantification in plasma turbulent simulations. It includes modules for loading and processing NetCDF and CSV datasets, estimating steady states, computing effective sample sizes, and running uncertainty quantification analyses. The project is structured into multiple Python scripts, each handling different aspects of the analysis. To see more information on documentation, etc... visit our website [here](https://sandialabs.github.io/quends/).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Detecting the Start of Statistical Steady State](#detecting-the-start-of-statistical-steady-state)
- [For Developers](#for-developers)
- [Documentation](#documentation)
- [Summary](#summary)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Install the package and dependencies**:
    You can install the package along with its dependencies using pip:
    ```bash
    pip install quends
    ```

2. **Verify the installation**:
    To ensure that the installation was successful, you can run a simple test:
    ```bash
    python -c "import quends; print('quends installed successfully')"
    ```


## Usage

Analyze a single data stream with the robust workflow, which detects the start
of statistical steady state (SSS) and computes statistics over it:

```python
import quends as qnds

ds = qnds.from_csv("examples/data/cgyro/output_nu0_50.csv", "Q_D/Q_GBD")
stats = qnds.RobustWorkflow().process_data_stream(ds, "Q_D/Q_GBD")

result = stats["Q_D/Q_GBD"]
print(result["mean"], result["mean_uncertainty"], result["sss_start"])
```

Loaders are available for CSV, NetCDF, JSON, NumPy arrays, dictionaries and GX
output (`from_csv`, `from_netcdf`, `from_json`, `from_numpy`, `from_dict`,
`from_gx`).

A small command-line interface prints a summary of one variable in a file:

```bash
quends summary examples/data/cgyro/output_nu0_50.csv Q_D/Q_GBD
```

### Examples
Tutorials are in [`examples/tutorial`](examples/tutorial), and use the shared
datasets in [`examples/data`](examples/data):
- `examples/tutorial/scripts/`: Python scripts that are also rendered in the
  [documentation gallery](https://sandialabs.github.io/quends/).
- `examples/tutorial/notebooks/`: Jupyter notebooks, including the DataStream
  guides (`DataStream_Guide*.ipynb`), the ensemble techniques (`03`-`06`), the
  robust workflow (`robust_workflow.ipynb`) and the stellarator analyses. Run
  them with `examples/tutorial/notebooks` as the working directory.

## Detecting the Start of Statistical Steady State

Finding where statistical steady state (SSS) starts is the hardest part of
the statistical analysis. All later statistics (mean, uncertainty, confidence
interval) use only the data after that point. If the start is too early,
transient data skews the results. If it is too late, data is thrown away and
the uncertainty goes up.

### Scope and choice of method

- The methods in QUENDS currently assume that a time series has **one (or no)
  initial transient, followed by one SSS segment that runs until the end of
  the time series**. Signals with several transients, or that drift away from
  SSS later on, are not supported.
- Most trimming strategies in QUENDS (`std`, `threshold`, `rolling_variance`,
  `self_consistent`, `iqr`, used via `DataStream.trim` or
  `build_trim_strategy`) only work well when there is a **very strong
  transient followed by well-behaved turbulence**.
- For time series with a small transient (or none), or with a lot of bursty
  turbulence, use the **`RobustWorkflow`**. Under the hood it uses the
  `mean_variation` heuristic (`MeanVariationTrimStrategy`).
- `mean_variation` is a heuristic. You will likely need to tune some of its
  hyperparameters for your data set. Once they are set, they generally work
  for the whole data set (for example, all runs of a parameter scan for the
  same quantity of interest).

### How the `mean_variation` approach works

Before it looks for SSS, `RobustWorkflow` checks that the signal is
stationary. If the signal is not stationary, it drops a fraction
(`drop_fraction`) of the data from the start and checks again. It then runs
the following steps on the remaining signal (see
`MeanVariationTrimStrategy.apply` in `src/quends/base/trim.py`):

1. **Decorrelation time.** It computes the decorrelation time of the signal
   (in number of samples) with `DataStream.compute_decorrelation_time()`.
2. **Smoothing.** It smooths the signal with a trailing rolling mean. The
   window is `decor_multiplier` times the decorrelation time (at least 3
   points, and no more than the signal length). This removes most of the
   turbulent fluctuations and keeps the slow trends, such as the transient.
3. **Noise level.** At each time point, it computes the standard deviation
   of the original signal from that point to the end. This measures the
   noise in the rest of the signal.
4. **Mean variation.** At each time point, it computes the mean of the
   remaining smoothed signal, from that point to the end. The *deviation* is
   the absolute difference between the smoothed signal at that point and this
   mean. Both the deviation and the standard deviation are smoothed a little
   more (`final_smoothing_window`), so they do not go to zero by construction
   at the end of the signal.
5. **Tolerance.** The tolerance is `std_dev_frac` times the noise level from
   step 3. A small term, `fudge_fac` times the magnitude of the initial mean,
   is added to the noise level so the tolerance never goes to zero. For
   signals whose mean magnitude is 1 or more, the tolerance is also scaled by
   the magnitude of the remaining mean.
6. **Criterion.** The criterion is met at the first point after which the
   deviation stays within the tolerance **for all remaining points**. If no
   such point exists, no SSS is found. In that case the workflow returns NaN
   results (`operate_safe=True`), or a rough "ball-park" estimate with a large
   uncertainty (`operate_safe=False`).
7. **Back correction.** The SSS start is placed
   `smoothing_window_correction × window` points *before* the point where the
   criterion was met.

**Why the back correction?** The smoothed value at a given time is a trailing
average over the preceding window. When the smoothed signal first meets the
criterion, the underlying signal has usually been in SSS for a while already:
the window has only just stopped averaging in transient data. Moving the start
back by the full window would usually go too far, into data that still has
some transient in it. So by default (`smoothing_window_correction=0.5`) the
start moves back by half the window.

The resulting SSS segment must also be at least `sss_time_min` long. If it is
shorter, the workflow treats it as if no SSS was found.

### Diagnosing and tuning

Set `verbosity=2` to plot the intermediate steps of the SSS detection:

```python
wf = qnds.RobustWorkflow(operate_safe=False, verbosity=2)
stats = wf.process_data_stream(ds, "Q_D/Q_GBD")
```

This shows the autocorrelation function with the decorrelation time, the
original and smoothed signal with the noise level, and, most importantly, the
**"Deviation and Tolerance vs. Time"** plot. That plot shows how far the
smoothed signal is from the mean of the remaining data, compared with the
noise-based tolerance. It also marks the point where the criterion is met and
the final SSS start after the back correction. (`verbosity=1` prints the same
information as text, without plots.)

Use these plots to adjust the hyperparameters of `RobustWorkflow`:

| Hyperparameter | Default | What it does / when to change it |
|---|---|---|
| `std_dev_frac` | 0.1 | **The most important one.** Sets the tolerance for variations in the mean of the SSS segment, as a fraction of the noise (std dev) in that segment. A spike or burst can shift the smoothed mean briefly and push the deviation above the tolerance. The criterion then only holds after the spike, so SSS starts too late. If so, **increase `std_dev_frac`**. The algorithm can then pick a much earlier start. Decrease it if SSS is detected while the signal is still visibly drifting. |
| `decor_multiplier` | 6.0 | Width of the smoothing window as a multiple of the decorrelation time. If the smoothed signal (and the deviation curve) is still quite noisy, **increase `decor_multiplier`** for a smoother mean. A wider window also makes the back correction larger, in number of points. |
| `smoothing_window_correction` | 0.5 | Fraction of the smoothing window by which the start moves back from the point where the criterion is met. If that point looks right but the returned SSS start is too early (e.g. still in the transient), **decrease `smoothing_window_correction`** so the start does not move back as far. Increase it if the start is clearly later than needed. |
| `fudge_fac` | 0.1 | Keeps the tolerance from going to zero for signals with very little noise. Rarely needs changing. |
| `final_smoothing_window` | 10 | Extra smoothing (in points) of the deviation and the noise level near the end of the signal. Rarely needs changing. |
| `sss_time_min` | 50 | Minimum duration of an accepted SSS segment (in time units of the data). |

A practical workflow: choose a few representative signals from your data set,
run them with `verbosity=2`, and tune `std_dev_frac` first, then
`decor_multiplier`, then `smoothing_window_correction`, until the SSS starts
look reasonable. Then process the full data set with `verbosity=0` and the
same settings. The notebook
[`examples/tutorial/notebooks/robust_workflow.ipynb`](examples/tutorial/notebooks/robust_workflow.ipynb)
walks through several examples, including ones where these hyperparameters
are changed.

## For Developers

1. **Clone the repository**:
    - Using SSH:
    ```bash
    git clone git@github.com:sandialabs/quends.git
    cd quends
    ```
    - Using HTTPS:
    ```bash
    git clone https://github.com/sandialabs/quends.git
    cd quends
    ```

2. **Install the package and dependencies**:
    The project uses [uv](https://docs.astral.sh/uv/) and a lockfile (`uv.lock`):
    ```bash
    uv sync --extra dev
    ```
    Alternatively, with pip: `pip install -e ".[dev]"`.

3. **Install pre-commit hooks**:
    The hooks run Black, isort and Ruff (in that order) on each commit:
    ```bash
    uv run pre-commit install
    ```

4. **Run the tests**:
    ```bash
    uv run pytest tests/
    uv run coverage run -m pytest tests/ && uv run coverage report
    ```
    Coverage must stay at or above 90%. Pull requests to `main` also fail if
    coverage drops at all compared to `main` (checked by Coveralls), so add tests
    for new code, and when removing code.

5. **Format and lint the files you changed**:
    ```bash
    uv run black <files>
    uv run isort --profile black <files>
    uv run ruff check --fix <files>
    ```
    The existing code base is not fully formatted yet, so avoid running these
    on the whole repository; that creates large unrelated diffs.

6. **Build the documentation (optional)**:
    ```bash
    uv run --with-requirements docs/requirements.txt sphinx-build -b html -W --keep-going docs docs/_build/html
    ```
    This regenerates tracked files under `docs/autoapi/` and
    `docs/auto_tutorials/`. Revert them unless you mean to update them.

## Developers: Publishing `quends` to PyPI
This section is for maintainers who publish new releases of `quends`.

Publishing is automated by the GitHub Actions workflow
[`.github/workflows/publish-to-pypi.yml`](.github/workflows/publish-to-pypi.yml).
It runs when a GitHub Release is **published** (not on pushes, tags, or draft
releases), builds the sdist and wheel, checks them with `twine check`, and
uploads them to PyPI using [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(no API token is stored in the repository).

1. **Update the version**
    Bump the version in **both** places (they must match, and PyPI rejects a
    version that has already been uploaded):
    - `version` in `pyproject.toml`
    - `__version__` in `src/quends/__init__.py`

    Then refresh the lockfile with `uv lock` and add an entry to `docs/changelog.rst`.
    List any breaking changes (removed or renamed arguments) there, and update
    the tutorials under `examples/tutorial/` that use them.

2. **Check the build locally (optional)**
    ```bash
    uv run pytest tests/
    uv build
    uvx twine check dist/*
    ```

3. **Merge to `main`**
    Make sure the `Run Tests` workflow passes on the commit you want to release.
    The publish workflow does not run the test suite itself.

4. **Create a GitHub Release**
    On GitHub, go to **Releases → Draft a new release**, create a new tag
    matching the version (e.g. `v0.1.4`) on `main`, add release notes, and click
    **Publish release**. The `Publish Python Package` workflow will then build
    and upload the package to PyPI. Progress can be followed in the **Actions** tab.

### Manual fallback
If the automated workflow cannot be used, a release can be uploaded by hand
(requires PyPI credentials for the `quends` project):
```bash
uv build
uvx twine upload --repository testpypi dist/*   # optional: test on TestPyPI first
pip install -i https://test.pypi.org/simple/ quends
uvx twine upload dist/*
```

## Documentation
For comprehensive information on how to use the QUENDS package, please refer to our [official documentation](https://sandialabs.github.io/quends/). The release history is in [`docs/changelog.rst`](docs/changelog.rst).

## Summary
Key functionalities include:
- **Data Handling**: Load and preprocess data from CSV, NetCDF, JSON, NumPy arrays, dictionaries and GX output.
- **Steady-State Detection**: Find and trim to the start of statistical steady state with several trimming strategies, or the automated `RobustWorkflow`.
- **Statistical Analysis**: Compute statistics, decorrelation times, effective sample sizes and confidence intervals, for single data streams and ensembles.
- **Visualization**: Create informative plots to visualize trends, correlations, and patterns in time series data.

## Contributing
Feel free to submit issues and merge requests. For major changes, please open an issue first to discuss what you would like to change.

## License
BSD 3-Clause License

Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

