# AGENTS.md

## Project

QUENDS (Quantification of Uncertainty in ENsembles of Data Streams) -- Python package for UQ in plasma turbulent simulations. BSD 3-Clause, Sandia National Laboratories.

## Setup

```bash
uv sync --extra dev          # install package + dev deps into .venv
pre-commit install           # enable Black, isort, Ruff hooks
```

Package manager is **uv** (lockfile: `uv.lock`). Do not use pip for dependency management. (CI itself installs with `pip install -e .[dev]`, so keep `pyproject.toml` the source of truth for dependencies.)

## Commands

| Task | Command |
|---|---|
| Run all tests | `uv run pytest tests/` |
| Run a single test file | `uv run pytest tests/test_data_stream.py` |
| Run a single test | `uv run pytest tests/test_data_stream.py::test_name -v` |
| Coverage (CI) | `uv run coverage run -m pytest tests/ && uv run coverage report` |
| Coverage, missing lines | `uv run coverage report -m` |
| Lint | `uv run ruff check --fix <files>` |
| Sort imports | `uv run isort --profile black <files>` |
| Format | `uv run black <files>` |
| Build docs | `uv run --with-requirements docs/requirements.txt sphinx-build -b html -W --keep-going docs docs/_build/html` |
| Build package | `uv build && uvx twine check dist/*` |
| CLI | `uv run quends summary <file.csv\|file.nc> <variable>` |

Pre-commit order: **black -> isort -> ruff**.

**Format only the files you change.** The repo is not fully Black/isort clean (`black --check .` currently flags ~20 files), and `ruff check src` reports many pre-existing findings. Running `black .` or `ruff check --fix` on the whole tree produces large unrelated diffs.

## Coverage

- Coverage must stay at or above **90%** (`fail_under = 90` in both `pyproject.toml` and `.coveragerc`). Currently ~92%.
- `src/quends/postprocessing/*` and `__init__.py` files are excluded.
- **CI also fails on any decrease relative to `main`.** `deployment.yml` uploads `coverage.xml` to Coveralls, and the PR check fails if coverage drops at all -- even by 0.007%. Deleting already-covered code lowers the percentage because the uncovered lines become a larger share of the total. When you remove code, add tests for some currently uncovered lines to compensate.
- To find where a change came from, run `coverage report` on the base branch (e.g. in a `git worktree`) and on your branch, then diff the per-file numbers.

## Source Layout

```
src/quends/
  base/           # Core: DataStream, Ensemble, History, operations, trim, stationary
  preprocessing/  # Loaders: csv, netcdf, json, numpy, dictionary, gx
  postprocessing/ # Exporter, loader, plotter, writer (excluded from coverage)
  workflow/       # High-level workflows: robust, batch_ensemble, ensemble_average, ensemble_statistics
  cli.py          # CLI entrypoint (`quends` console script, `python -m quends`)
  __main__.py     # `python -m quends` entry point
```

Package uses `src` layout (`[tool.setuptools] package-dir = {"" = "src"}`). Imports are `from quends.base.data_stream import ...`, not `from src.quends...`. Public API is re-exported from `src/quends/__init__.py`.

### API notes

- `MeanVariationTrimStrategy` (SSS detection, used by `RobustWorkflow`) takes `verbosity`, `decor_multiplier`, `std_dev_frac`, `fudge_fac`, `smoothing_window_correction`, `final_smoothing_window`. The decorrelation time comes from `DataStream.compute_decorrelation_time()`. `max_lag_frac` and `autocorr_sig_level` were removed in 0.1.4 and now raise `TypeError`.
- Removing or renaming public arguments is a breaking change. Record it in `docs/changelog.rst`, and update the notebooks/scripts under `examples/tutorial/` that use it.

## Tests

- ~525 tests, all in `tests/`, flat file structure mirroring source modules.
- Shared fixtures live in `tests/_shared.py`. There is no `conftest.py`; files that need the shared fixtures declare `pytest_plugins = ("tests._shared",)`.
- `pyproject.toml` sets `filterwarnings = ["ignore"]`, so warnings are hidden by default. Use `-W error::<Category>` to surface them.
- Test data directories (`tests/cgyro/`, `tests/guide/`, `tests/robust_workflow/`, `tests/tutorial/`) contain `expected/` CSV files for regression testing and `output/` for generated artifacts.
- Running the tests rewrites the tracked files under `tests/*/output/`. Do not commit them unless you are intentionally updating them. If a numerical change is intended, regenerate the matching `expected/` CSVs.
- Some tests execute the tutorial notebooks with papermill.

## Docs

- Sphinx sources are in `docs/`; `docs/changelog.rst` is the changelog.
- A docs build regenerates the **tracked** files under `docs/autoapi/` and `docs/auto_tutorials/` (from `examples/tutorial/scripts/`) and touches `docs/sg_execution_times.rst`. Revert them (`git checkout -- docs/autoapi docs/auto_tutorials docs/sg_execution_times.rst`) unless you intend to commit regenerated docs.
- The `No module named 'pkg_resources'` message from `docs/conf.py` is caught and harmless.

## Style

- Formatter: **Black** (default settings)
- Import sorting: **isort** with `profile = "black"` (`.isort.cfg`)
- Linter: **Ruff** (no project config file)
- Python version: `>=3.8` declared, CI uses 3.9. Avoid syntax/APIs newer than 3.9 in `src/`.

## CI

Three GitHub Actions workflows:
- `python-tests.yml` -- runs `pytest tests/` on push to any branch and on PRs to `main`
- `deployment.yml` -- on PRs to `main` and pushes to `main`: builds Sphinx docs (`-W`, warnings are errors), runs coverage, uploads to Coveralls; deploys to GitHub Pages only on push to `main`
- `publish-to-pypi.yml` -- builds and publishes to PyPI when a GitHub Release is published

## Releases

Follow "Developers: Publishing `quends` to PyPI" in `README.md`. In short:
1. Bump the version in **both** `pyproject.toml` and `src/quends/__init__.py`.
2. Run `uv lock`.
3. Add a `docs/changelog.rst` entry.
4. Merge to `main` with green CI.
5. Publish a GitHub Release tagged `vX.Y.Z`.

## Container Dev Environment

`./dev.litellm` builds and runs a Podman container (`Containerfile.litellm`, Python 3.12) with uv and OpenCode pre-installed. Both files are gitignored (local only). It needs:
- `OPENAI_API_KEY` and `OPENAI_BASE_URL`
- `DEV_CA_BUNDLE` (a PEM bundle), except on macOS, where one is generated from the system keychain.

It uses a named Podman volume for `.venv` persistence and runs `uv sync --locked` before starting.
