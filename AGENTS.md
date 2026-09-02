# AGENTS.md

## Project

QUENDS (Quantification of Uncertainty in ENsembles of Data Streams) -- Python package for UQ in plasma turbulent simulations. BSD 3-Clause, Sandia National Laboratories.

## Setup

```bash
uv sync --extra dev          # install package + dev deps into .venv
pre-commit install           # enable Black, isort, Ruff hooks
```

Package manager is **uv** (lockfile: `uv.lock`). Do not use pip for dependency management.

## Commands

| Task | Command |
|---|---|
| Run all tests | `uv run pytest tests/` |
| Run a single test file | `uv run pytest tests/test_data_stream.py` |
| Run a single test | `uv run pytest tests/test_data_stream.py::test_name -v` |
| Coverage (CI) | `uv run coverage run -m pytest tests/ && uv run coverage report` |
| Lint | `uv run ruff check --fix` |
| Sort imports | `uv run isort . --profile black` |
| Format | `uv run black .` |
| Build docs | `uv run sphinx-build -b html -W --keep-going docs docs/_build/html` |

Pre-commit order: **black -> isort -> ruff**. Run all three before committing.

## Coverage

Coverage must stay at or above **90%** (`fail_under = 90` in both `pyproject.toml` and `.coveragerc`). `src/quends/postprocessing/*` is excluded from coverage. Current coverage is at exactly 90% -- any new code in measured modules needs tests.

## Source Layout

```
src/quends/
  base/           # Core: DataStream, Ensemble, History, operations, trim, stationary
  preprocessing/  # Loaders: csv, netcdf, json, numpy, dictionary, gx
  postprocessing/ # Exporter, loader, plotter, writer (excluded from coverage)
  workflow/       # High-level workflows: robust, batch_ensemble, ensemble_average, ensemble_statistics
  cli.py          # CLI entrypoint (quends command)
```

Package uses `src` layout (`[tool.setuptools] package-dir = {"" = "src"}`). Imports are `from quends.base.data_stream import ...`, not `from src.quends...`.

## Tests

- 479 tests, all in `tests/`, flat file structure mirroring source modules.
- Shared fixtures live in `tests/_shared.py` (not a conftest.py -- fixtures must be imported explicitly or referenced via pytest's fixture discovery).
- Test data directories (`tests/cgyro/`, `tests/guide/`, `tests/robust_workflow/`, `tests/tutorial/`) contain `expected/` CSV files for regression testing and `output/` for generated artifacts.
- No conftest.py exists; tests rely on per-file fixture definitions and `_shared.py`.

## Style

- Formatter: **Black** (default settings)
- Import sorting: **isort** with `profile = "black"` (`.isort.cfg`)
- Linter: **Ruff** (default rules, no custom config file)
- Python version: `>=3.8` declared, CI uses 3.9

## CI

Three GitHub Actions workflows:
- `python-tests.yml` -- runs `pytest tests/` on every push
- `deployment.yml` -- builds Sphinx docs, runs coverage, deploys to GitHub Pages on push to main
- `publish-to-pypi.yml` -- publishes to PyPI on GitHub release

## Container Dev Environment

`./dev.sandia` builds and runs a Podman container (`Containerfile.sandia`) with OpenCode pre-installed. Requires `OPENAI_API_KEY` and `OPENAI_BASE_URL` env vars. Uses a named Podman volume for `.venv` persistence.
