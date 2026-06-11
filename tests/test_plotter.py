# Tests for the Plotter uniform contract (return values, save/overwrite, precompute).
import os

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
    return Ensemble([
        DataStream(pd.DataFrame({"time": np.arange(300.0), "x": 5 + 0.2 * rng.standard_normal(300)}))
        for _ in range(3)
    ])


def test_trace_plot_returns_list_of_fig_axes(ds):
    res = ds  # noqa
    out = Plotter(output_dir=".").trace_plot(ds)
    assert isinstance(out, list)
    fig, axes = out[0]
    assert fig is not None


def test_steady_state_auto_saves_and_guards_overwrite(ds, tmp_path):
    p = Plotter(output_dir=str(tmp_path))
    out = p.steady_state_automatic_plot(ds, method="threshold", threshold=0.1,
                                        batch_size=20, save=True)
    assert os.path.exists(tmp_path / "steady_state_auto_Datastream.png")
    fig, axes = out[0]
    assert fig is not None
    with pytest.raises(FileExistsError):
        p.steady_state_automatic_plot(ds, method="threshold", threshold=0.1,
                                      batch_size=20, save=True)


def test_precomputed_ss_starts_skips_trim(ds):
    # Supplying ss_starts must render without invoking the trim machinery.
    out = Plotter(output_dir=".").steady_state_automatic_plot(ds, ss_starts={"x": 80.0})
    assert len(out) == 1


def test_plot_ensemble_with_average_precomputed_avg(ens, tmp_path):
    avg = ens.compute_average_ensemble().data
    fig, axes = Plotter(output_dir=str(tmp_path)).plot_ensemble_with_average(
        ens, avg_df=avg, save=True, filename="ea.png"
    )
    assert os.path.exists(tmp_path / "ea.png")
    assert fig is not None


def test_show_false_by_default_does_not_block(ds):
    # Should return immediately without entering an interactive show loop.
    out = Plotter(output_dir=".").trace_plot(ds)  # show defaults False
    assert out is not None
