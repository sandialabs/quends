from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quends import (
    DataStream,
    MeanVariationTrimStrategy,
    NoiseThresholdTrimStrategy,
    QuantileTrimStrategy,
    RollingVarianceThresholdTrimStrategy,
    TrimDataStreamOperation,
)

pytest_plugins = ("tests._shared",)


def make_sss_strategy(*, verbosity: int = 1):
    return MeanVariationTrimStrategy(
        max_lag_frac=0.5,
        verbosity=verbosity,
        autocorr_sig_level=0.05,
        decor_multiplier=4.0,
        std_dev_frac=0.1,
        fudge_fac=0.1,
        smoothing_window_correction=0.3,
        final_smoothing_window=10,
    )


def test_trim_std(trim_data: pd.DataFrame):
    ds = DataStream(trim_data)
    strategy = QuantileTrimStrategy(window_size=1, start_time=3.0, robust=True)
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert isinstance(result, DataStream)
    assert result.data.empty
    assert list(result.data.columns) == ["time", "A"]
    assert result._history == [
        {"operation": "is_stationary", "options": {"columns": "A"}},
        {
            "operation": "trim",
            "options": {
                "column_name": "A",
                "window_size": 1,
                "start_time": 3.0,
                "method": "std",
                "robust": True,
                "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
            },
        },
    ]


def test_trim_threshold(trim_data: pd.DataFrame):
    ds = DataStream(trim_data.astype(float))
    strategy = NoiseThresholdTrimStrategy(window_size=1, start_time=3.0, threshold=4)
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert isinstance(result, DataStream)
    assert result.data.empty
    assert list(result.data.columns) == ["time", "A"]
    assert result._history == [
        {"operation": "is_stationary", "options": {"columns": "A"}},
        {
            "operation": "trim",
            "options": {
                "column_name": "A",
                "window_size": 1,
                "start_time": 3.0,
                "method": "threshold",
                "threshold": 4,
                "robust": True,
                "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
            },
        },
    ]


def test_trim_rolling_variance(trim_data: pd.DataFrame):
    ds = DataStream(trim_data)
    strategy = RollingVarianceThresholdTrimStrategy(
        window_size=1, start_time=3.0, threshold=4
    )
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert isinstance(result, DataStream)
    assert result.data.empty
    assert list(result.data.columns) == ["time", "A"]
    assert result._history == [
        {"operation": "is_stationary", "options": {"columns": "A"}},
        {
            "operation": "trim",
            "options": {
                "column_name": "A",
                "window_size": 1,
                "start_time": 3.0,
                "method": "rolling_variance",
                "threshold": 4,
                "robust": True,
                "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
            },
        },
    ]


def test_sss_start_strategy_accepts_explicit_args(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    strategy = MeanVariationTrimStrategy(
        max_lag_frac=0.5,
        verbosity=0,
        autocorr_sig_level=0.05,
        decor_multiplier=4.0,
        std_dev_frac=0.1,
        fudge_fac=0.1,
        smoothing_window_correction=0.8,
        final_smoothing_window=10,
    )
    assert strategy.max_lag_frac == 0.5
    assert strategy.decor_multiplier == 4.0
    assert strategy.final_smoothing_window == 10
    trim_op = TrimDataStreamOperation(strategy=strategy)
    result = trim_op(ds, column_name="A")
    assert isinstance(result, (DataStream, pd.DataFrame))


def test_trim_missing_threshold(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    strategy = NoiseThresholdTrimStrategy()
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert isinstance(result, DataStream)
    assert result.data.empty
    assert set(result.data.columns) == {"time", "A", "B"}
    assert result._history == [
        {"operation": "is_stationary", "options": {"columns": "A"}},
        {
            "operation": "trim",
            "options": {
                "column_name": "A",
                "window_size": 10,
                "start_time": 0.0,
                "method": "threshold",
                "threshold": None,
                "robust": True,
                "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
            },
        },
    ]


def test_find_steady_state_std(trim_data: pd.DataFrame):
    strategy = QuantileTrimStrategy(window_size=1)
    assert strategy._detection_method(data=trim_data, column_name="A") == 0


def test_find_steady_state_std_non_robust(trim_data: pd.DataFrame):
    strategy = QuantileTrimStrategy(window_size=2, robust=False)
    assert strategy._detection_method(data=trim_data, column_name="A") == 3


def test_find_steady_state_not_valid(no_valid_data: pd.DataFrame):
    strategy = QuantileTrimStrategy(window_size=1)
    result = strategy._detection_method(data=no_valid_data, column_name=["time", "A"])
    assert result is None


def test_find_steady_state_stationary(stationary_data: pd.DataFrame):
    strategy = NoiseThresholdTrimStrategy(window_size=2, threshold=0.1)
    assert strategy._detection_method(data=stationary_data, column_name="A") == 1


def test_find_steady_state_long_data(long_data: pd.DataFrame):
    strategy = NoiseThresholdTrimStrategy(window_size=2, threshold=0.1)
    assert strategy._detection_method(data=long_data, column_name="A") is None


def test_find_steady_state_trim_data(trim_data: pd.DataFrame):
    strategy = NoiseThresholdTrimStrategy(window_size=3, threshold=0.5)
    assert strategy._detection_method(data=trim_data, column_name="A") == 2


def test_find_steady_state_with_start_time(long_data: pd.DataFrame):
    DataStream(long_data)


def test_find_steady_state_rolling_variance_stationary(stationary_data: pd.DataFrame):
    strategy = RollingVarianceThresholdTrimStrategy(window_size=3)
    assert strategy._detection_method(data=stationary_data, column_name="A") is None


def test_find_steady_state_none_rolling_variance(long_data: pd.DataFrame):
    strategy = RollingVarianceThresholdTrimStrategy(window_size=3, threshold=0.1)
    assert strategy._detection_method(data=long_data, column_name="A") is None


def test_find_steady_state_rolling_variance_not_valid(no_valid_data: pd.DataFrame):
    strategy = RollingVarianceThresholdTrimStrategy(window_size=1)
    assert strategy._detection_method(data=no_valid_data, column_name="A") is None


def test_trim_sss_start_detects_sss(slope_to_stationary_df: pd.DataFrame):
    ds = DataStream(slope_to_stationary_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, DataStream)
    assert not trimmed.data.empty
    assert len(trimmed.data) < len(ds.data)
    assert trimmed.data["time"].iloc[0] > 20


def test_trim_sss_start_already_stationary(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, pd.DataFrame)
    assert trimmed.empty


def test_trim_sss_start_no_sss_found(persistent_trend_df: pd.DataFrame):
    ds = DataStream(persistent_trend_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, DataStream)
    assert not trimmed.data.empty


def test_trim_sss_start_verbose_output(
    slope_to_stationary_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
):
    ds = DataStream(slope_to_stationary_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy(verbosity=2))

    trimmed = trim_op(ds, column_name="A")
    captured = capsys.readouterr()

    assert isinstance(trimmed, DataStream)
    assert (
        "Getting start of SSS" in captured.out
        or "criterion is met" in captured.out
        or "start of SSS" in captured.out
    )


def test_trim_sss_start_handles_nan_values(stationary_noise_df: pd.DataFrame):
    df = stationary_noise_df.copy()
    df.loc[50:80, "A"] = np.nan
    ds = DataStream(df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert trimmed is not None


def test_trim_sss_start_intermittent_spikes(intermittent_stationary_df: pd.DataFrame):
    ds = DataStream(intermittent_stationary_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert trimmed is not None


def test_trim_sss_start_high_frequency_noise(high_frequency_noise_df: pd.DataFrame):
    ds = DataStream(high_frequency_noise_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert trimmed is not None


def test_trim_sss_start_decaying_oscillation(oscillating_to_stable_df: pd.DataFrame):
    ds = DataStream(oscillating_to_stable_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, DataStream)
    assert not trimmed.data.empty
    assert len(trimmed.data) < len(ds.data)
    assert trimmed.data["time"].iloc[0] > 100


def test_trim_sss_start_multiple_transitions(multiple_transitions_df: pd.DataFrame):
    ds = DataStream(multiple_transitions_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, DataStream)
    assert not trimmed.data.empty
    assert trimmed.data["time"].iloc[0] > 150


def test_trim_sss_start_oscillation_trims_correctly(
    oscillating_to_stable_df: pd.DataFrame,
):
    ds = DataStream(oscillating_to_stable_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert len(trimmed.data) < len(ds.data) * 0.8
    assert hasattr(trimmed, "data")


def test_trim_sss_start_intermittent_has_spikes(
    intermittent_stationary_df: pd.DataFrame,
):
    ds = DataStream(intermittent_stationary_df)
    large_values = ds.data["A"] > 20
    assert large_values.sum() >= 15


def test_trim_sss_start_high_freq_oscillates(high_frequency_noise_df: pd.DataFrame):
    ds = DataStream(high_frequency_noise_df)
    detrended = ds.data["A"] - ds.data["A"].mean()
    zero_crossings = np.sum(np.diff(np.sign(detrended)) != 0)
    assert zero_crossings > 50


def test_trim_sss_start_returns_datastream_or_dataframe(
    oscillating_to_stable_df: pd.DataFrame,
):
    ds = DataStream(oscillating_to_stable_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy())

    trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, (DataStream, pd.DataFrame))


def test_trim_sss_start_verbose_plotting_no_sss(
    intermittent_stationary_df: pd.DataFrame,
):
    ds = DataStream(intermittent_stationary_df)
    trim_op = TrimDataStreamOperation(strategy=make_sss_strategy(verbosity=2))

    trimmed = trim_op(ds, column_name="A")

    assert trimmed is not None


def test_trim_sss_start_verbose_plotting_runs_without_error(
    oscillating_to_stable_df: pd.DataFrame,
):
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"):
        ds = DataStream(oscillating_to_stable_df)
        trim_op = TrimDataStreamOperation(strategy=make_sss_strategy(verbosity=2))
        trimmed = trim_op(ds, column_name="A")

    assert isinstance(trimmed, DataStream)
    assert not trimmed.data.empty
