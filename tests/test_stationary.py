from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quends import DataStream, MakeDataStreamStationaryOperation

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
    return MakeDataStreamStationaryOperation(
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
    op = MakeDataStreamStationaryOperation(
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


def test_make_stationary_retries_until_stationary(
    capsys: pytest.CaptureFixture[str],
):
    ds = DataStream(pd.DataFrame({"time": range(10), "A": range(10)}))
    op = make_stationary_op(
        column="A",
        n_pts_orig=len(ds.data),
        verbosity=1,
        drop_fraction=0.2,
        n_pts_min=1,
        n_pts_frac_min=0.0,
    )

    with patch.object(
        DataStream,
        "is_stationary",
        side_effect=[{"A": False}, {"A": False}, {"A": True}],
    ):
        result_ds, stationary = op(ds)

    captured = capsys.readouterr()

    assert stationary is True
    assert len(result_ds.data) == 7
    assert "not stationary, even after dropping first 2 points." in captured.out
    assert "is stationary after dropping first 3 points." in captured.out


def test_make_stationary_loops_multiple_times(
    capsys: pytest.CaptureFixture[str],
):
    df = pd.DataFrame({"time": range(20), "A": range(20)})
    ds = DataStream(df)
    op = make_stationary_op(
        column="A",
        n_pts_orig=len(ds.data),
        verbosity=1,
        drop_fraction=0.1,
        n_pts_min=1,
        n_pts_frac_min=0.0,
    )

    call_count = 0

    def fake_is_stationary(self, columns):
        nonlocal call_count
        call_count += 1
        col = columns[0] if isinstance(columns, list) else columns
        return {col: call_count >= 3}

    with patch.object(DataStream, "is_stationary", fake_is_stationary):
        result_ds, stationary = op._apply(ds)

    captured = capsys.readouterr()
    assert call_count >= 3
    assert "not stationary, even after dropping first" in captured.out
    assert "is stationary after dropping first" in captured.out
