"""Tests for the ensemble (and mean-overlay) Plotter methods."""

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from quends import DataStream, Ensemble, Plotter  # noqa: E402


@pytest.fixture
def ds():
    rng = np.random.default_rng(0)
    y = np.r_[np.linspace(0, 5, 80), 5 + 0.2 * rng.standard_normal(320)]
    return DataStream(pd.DataFrame({"time": np.arange(400.0), "x": y}))


@pytest.fixture
def ens():
    rng = np.random.default_rng(1)
    return Ensemble(
        [
            DataStream(
                pd.DataFrame(
                    {"time": np.arange(300.0), "x": 5 + 0.2 * rng.standard_normal(300)}
                )
            )
            for _ in range(3)
        ]
    )


def _assert_fig(out):
    """Methods return either a list of (fig, axes) or a single (fig, axes)."""
    if isinstance(out, list):
        assert out, "expected a non-empty list of (fig, axes)"
        fig, _ = out[0]
    else:
        fig, _ = out
    assert fig is not None


def test_trace_plot_with_mean(ds):
    _assert_fig(Plotter().trace_plot_with_mean(ds, ["x"]))


def test_ensemble_trace_plot(ens):
    _assert_fig(Plotter().ensemble_trace_plot(ens, ["x"]))


def test_ensemble_trace_plot_with_mean(ens):
    _assert_fig(Plotter().ensemble_trace_plot_with_mean(ens, ["x"]))


def test_plot_acf_ensemble(ens):
    assert Plotter().plot_acf_ensemble(ens, column="x") is not None


def test_ensemble_steady_state_automatic_plot(ens):
    _assert_fig(
        Plotter().ensemble_steady_state_automatic_plot(
            ens, ["x"], method="std", batch_size=20
        )
    )


def test_ensemble_steady_state_plot(ens):
    _assert_fig(
        Plotter().ensemble_steady_state_plot(ens, ["x"], steady_state_start=50.0)
    )


def test_plot_ensemble(ens):
    _assert_fig(Plotter().plot_ensemble(ens, ["x"]))


def test_plot_ensemble_with_average(ens):
    _assert_fig(Plotter().plot_ensemble_with_average(ens, ["x"]))
