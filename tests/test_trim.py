from unittest.mock import MagicMock, patch

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


#  _check_stationary fallback


def test_check_stationary_returns_false_for_non_dict():
    """Covers the `return False` branch when is_stationary doesn't return a dict."""
    ds = MagicMock(spec=DataStream)
    ds.is_stationary.return_value = True  # not a dict
    strategy = QuantileTrimStrategy()
    assert strategy._check_stationary(ds, "A") is False


def test_check_stationary_returns_false_when_column_not_in_dict():
    ds = MagicMock(spec=DataStream)
    ds.is_stationary.return_value = {"B": True}  # "A" missing
    strategy = QuantileTrimStrategy()
    assert strategy._check_stationary(ds, "A") is False


#  _preprocess


def test_preprocess_strips_leading_zeros():
    df = pd.DataFrame(
        {
            "time": np.arange(10.0),
            "A": [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    ds = MagicMock(spec=DataStream)
    ds.data = df
    strategy = QuantileTrimStrategy(start_time=0.0)
    result = strategy._preprocess(ds, "A")
    assert result["A"].iloc[0] > 0


def test_preprocess_filters_by_start_time():
    df = pd.DataFrame(
        {
            "time": np.arange(10.0),
            "A": np.ones(10),
        }
    )
    ds = MagicMock(spec=DataStream)
    ds.data = df
    strategy = QuantileTrimStrategy(start_time=5.0)
    result = strategy._preprocess(ds, "A")
    assert result["time"].min() >= 5.0


def test_preprocess_no_leading_zeros_unchanged():
    """If no leading zeros, data should not be further trimmed."""
    df = pd.DataFrame(
        {
            "time": np.arange(5.0),
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ds = MagicMock(spec=DataStream)
    ds.data = df
    strategy = QuantileTrimStrategy(start_time=0.0)
    result = strategy._preprocess(ds, "A")
    assert len(result) == 5


#  Missing threshold error message


def test_trim_missing_threshold_error_message(long_stationary_data: pd.DataFrame):
    """
    Hits the 'Threshold must be specified' branch, which requires
    stationarity to pass first.
    """
    ds = MagicMock(spec=DataStream)
    ds.data = long_stationary_data
    ds.history = []
    ds.is_stationary.return_value = {"A": True}
    strategy = NoiseThresholdTrimStrategy(threshold=None)
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert result.data.empty
    assert "Threshold must be specified" in result.message


# TrimDataStreamOperation: strategy with no `robust` attribute should not include it in history options


def test_trim_std_success_path(long_stationary_data: pd.DataFrame):
    """Tests the success branch of QuantileTrimStrategy through TrimDataStreamOperation."""
    df = pd.DataFrame({"time": np.arange(100.0), "A": np.full(100, 5.0)})
    ds = MagicMock(spec=DataStream)
    ds.data = df
    ds.history = []
    ds.is_stationary.return_value = {"A": True}
    strategy = QuantileTrimStrategy(window_size=10, robust=True)
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert isinstance(result, DataStream)
    assert not result.data.empty
    assert result.data["time"].iloc[0] == 0.0
    assert "message" not in result._history[1]["options"]


def test_trim_rolling_variance_success_path(long_stationary_data: pd.DataFrame):
    rng = np.random.default_rng(0)
    signal = np.concatenate([rng.normal(5, 2, 50), rng.normal(5, 0.1, 50)])
    df = pd.DataFrame({"time": np.arange(100.0), "A": signal})
    ds = MagicMock(spec=DataStream)
    ds.data = df
    ds.history = []
    ds.is_stationary.return_value = {"A": True}
    strategy = RollingVarianceThresholdTrimStrategy(window_size=10, threshold=0.9)
    trim_op = TrimDataStreamOperation(strategy=strategy)

    result = trim_op(ds, column_name="A")

    assert isinstance(result, DataStream)
    assert not result.data.empty
    assert result.data["time"].iloc[0] > 0.0


# TrimDataStreamOperation: strategy with no `robust` attribute should not include it in history options


def test_trim_operation_history_excludes_robust_when_absent():
    """Strategies without a `robust` attr should not have it in history options."""
    signal = np.ones(100)
    df = pd.DataFrame({"time": np.arange(100.0), "A": signal})
    ds = MagicMock(spec=DataStream)
    ds.data = df
    ds.history = []
    ds.is_stationary.return_value = {"A": False}

    strategy = RollingVarianceThresholdTrimStrategy(window_size=10, threshold=0.5)
    # Manually remove robust to simulate a strategy without it
    if hasattr(strategy, "robust"):
        delattr(strategy, "robust")

    trim_op = TrimDataStreamOperation(strategy=strategy)
    result = trim_op(ds, column_name="A")
    trim_options = result._history[1]["options"]

    assert "robust" not in trim_options


# MeanVariationTrimStrategy


def _make_no_sss_where_side_effect():
    original_where = np.where
    call_count = {"count": 0}

    def side_effect(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 2:
            return (np.array([], dtype=int),)
        return original_where(*args, **kwargs)

    return side_effect


def test_trim_sss_no_sss_else_branch_returns_empty_dataframe(
    persistent_trend_df: pd.DataFrame,
):
    """
    When no SSS is found, trimmed_stream is a plain pd.DataFrame
    with columns ['time', 'flux'] (hardcoded in the else branch).
    """
    ds = DataStream(persistent_trend_df)
    strategy = make_sss_strategy(verbosity=0)

    with patch(
        "quends.base.trim.np.where", side_effect=_make_no_sss_where_side_effect()
    ):
        result = strategy.apply(ds, column_name="A")

    # The no-SSS path returns a raw DataFrame, not a DataStream
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["time", "flux"]
    assert result.empty


def test_trim_sss_no_sss_verbosity_1_prints_message(
    persistent_trend_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
):
    """
    verbosity=1 in the no-SSS branch should print the 'No SSS found' message
    but NOT trigger the plotting block (verbosity > 1).
    """
    ds = DataStream(persistent_trend_df)
    strategy = make_sss_strategy(verbosity=1)

    with patch(
        "quends.base.trim.np.where", side_effect=_make_no_sss_where_side_effect()
    ):
        strategy.apply(ds, column_name="A")
    captured = capsys.readouterr()

    assert "No SSS found" in captured.out


def test_trim_sss_no_sss_verbosity_0_no_output(
    persistent_trend_df: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
):
    """verbosity=0 should produce no stdout at all."""
    ds = DataStream(persistent_trend_df)
    strategy = make_sss_strategy(verbosity=0)

    with patch(
        "quends.base.trim.np.where", side_effect=_make_no_sss_where_side_effect()
    ):
        strategy.apply(ds, column_name="A")
    captured = capsys.readouterr()

    assert captured.out == ""


def test_trim_sss_no_sss_verbosity_2_triggers_plot(
    persistent_trend_df: pd.DataFrame,
):
    """
    verbosity=2 in the no-SSS else branch should call plt.show()
    for the deviation/tolerance plot. This covers line 644->645.
    """
    with patch("matplotlib.pyplot.show") as mock_show, patch(
        "matplotlib.pyplot.figure"
    ), patch("matplotlib.pyplot.plot"), patch("matplotlib.pyplot.close"):

        ds = DataStream(persistent_trend_df)
        strategy = make_sss_strategy(verbosity=2)
        with patch(
            "quends.base.trim.np.where", side_effect=_make_no_sss_where_side_effect()
        ):
            strategy.apply(ds, column_name="A")

    # plt.show() should have been called at least once for the no-SSS plot
    assert mock_show.call_count >= 1


def test_trim_sss_no_sss_verbosity_2_plots_deviation_and_tolerance(
    persistent_trend_df: pd.DataFrame,
):
    """
    Checks that both deviation and tolerance are plotted in the no-SSS branch.
    """
    plot_calls = []

    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"), patch(
        "matplotlib.pyplot.plot",
        side_effect=lambda *a, **kw: plot_calls.append(kw.get("label")),
    ), patch("matplotlib.pyplot.close"):

        ds = DataStream(persistent_trend_df)
        strategy = make_sss_strategy(verbosity=2)
        with patch(
            "quends.base.trim.np.where", side_effect=_make_no_sss_where_side_effect()
        ):
            strategy.apply(ds, column_name="A")

    # The no-SSS plot draws Deviation and Tolerance
    assert "Deviation" in plot_calls
    assert "Tolerance" in plot_calls
