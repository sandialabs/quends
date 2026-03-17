import numpy as np
import pandas as pd
import pytest

from quends import DataStream, MakeStationaryOperation

pytest_plugins = ("tests._shared",)


def make_stationary_op(
    column: str,
    n_pts_orig: int,
    *,
    operate_safe: bool = False,
    verbosity: int = 1,
    drop_fraction: float = 0.2,
    n_pts_min: int = 50,
    n_pts_frac_min: float = 0.2,
):
    return MakeStationaryOperation(
        column=column,
        n_pts_orig=n_pts_orig,
        operate_safe=operate_safe,
        verbosity=verbosity,
        drop_fraction=drop_fraction,
        n_pts_min=n_pts_min,
        n_pts_frac_min=n_pts_frac_min,
    )


def test_is_stationary(stationary_data: pd.DataFrame):
    ds = DataStream(stationary_data)
    assert ds.is_stationary(columns="A") == {"A": "Error: Invalid input, x is constant"}


def test_is_not_stationary(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    out = ds.is_stationary(columns="A")
    if hasattr(np, "False_"):
        assert out == {"A": np.False_}
    else:
        assert out == {"A": False}


def test_make_stationary_operation_accepts_explicit_args(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    op = MakeStationaryOperation(
        column="A",
        n_pts_orig=len(ds.data),
        operate_safe=True,
        n_pts_min=100,
        n_pts_frac_min=0.2,
        drop_fraction=0.25,
        verbosity=0,
    )
    result, stationary = op(ds)
    assert not stationary
    assert isinstance(result, DataStream)


def test_make_stationary_with_stationary_data(stationary_data: pd.DataFrame):
    ds = DataStream(stationary_data)
    op = make_stationary_op(column="A", n_pts_orig=len(stationary_data))

    result_ds, stationary = op(ds)

    assert stationary == "Error: Invalid input, x is constant"
    assert len(result_ds.data) == len(stationary_data)


def test_make_stationary_already_stationary(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    op = make_stationary_op(column="A", n_pts_orig=len(ds.data))

    result_ds, stationary = op(ds)

    assert stationary.any() == np.True_
    assert len(result_ds.data) == len(ds.data)


def test_make_stationary_drops_trend(slope_to_stationary_df: pd.DataFrame):
    ds = DataStream(slope_to_stationary_df)
    op = make_stationary_op(column="A", n_pts_orig=len(ds.data))

    result_ds, stationary = op(ds)

    assert stationary == np.True_
    assert len(result_ds.data) < len(ds.data)


def test_make_stationary_verbose_output(
    slope_to_stationary_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
):
    ds = DataStream(slope_to_stationary_df)
    op = make_stationary_op(column="A", n_pts_orig=len(ds.data))

    op(ds)
    captured = capsys.readouterr()

    assert "stationary after dropping first" in captured.out


def test_make_stationary_verbose_output_fails(
    persistent_trend_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
):
    ds = DataStream(persistent_trend_df)
    op = make_stationary_op(column="A", n_pts_orig=len(ds.data))

    result_ds, stationary = op(ds)
    captured = capsys.readouterr()

    assert "not stationary" in captured.out
    assert result_ds.data.empty
    assert stationary is False or stationary == np.False_
