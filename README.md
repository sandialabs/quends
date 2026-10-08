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

