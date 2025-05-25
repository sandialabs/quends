# Import statements
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

# Special imports
from quends import DataStream


# Fixtures
@pytest.fixture
def empty_data():
    return pd.DataFrame()


@pytest.fixture
def simple_data():
    return pd.DataFrame({"A": [1, 2, 3]})


@pytest.fixture
def long_data():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
        }
    )


@pytest.fixture
def trim_data():
    return pd.DataFrame(
        {"time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    )


@pytest.fixture
def nan_data():
    return pd.DataFrame({"A": [None, None, None]})


# Test DataStream initialization
# =============================================================================


# Test initialization with simple data set
def test_init_simple(simple_data):
    ds = DataStream(simple_data)
    assert len(ds) == 3
    assert "A" in ds.variables()


# Test initialization with empty data set
def test_init_empty(empty_data):
    ds = DataStream(empty_data)  # Does not throw an error
    assert len(ds) == 0
    assert len(ds.variables()) == 0


# Test mean
# =============================================================================


# Test mean with simple data set
def test_mean_simple(simple_data):
    ds = DataStream(simple_data)
    mean = ds.mean(window_size=1)
    expected = {"A": {"mean": 2.0}}
    assert mean == expected


# Test mean with empty data set
def test_mean_empty(empty_data):
    ds = DataStream(empty_data)
    mean = ds.mean()
    expected = {}
    assert mean == expected


# Test mean with long
def test_mean_long(long_data):
    ds = DataStream(long_data)
    mean = ds.mean()
    expected = expected = {"A": {"mean": 3.0}, "B": {"mean": 3.0}}
    assert mean == expected


# Test mean with overlapping windows
def test_mean_long_overlapping_window(long_data):
    ds = DataStream(long_data)
    mean = ds.mean()
    expected = {"A": {"mean": 3.0}, "B": {"mean": 3.0}}
    print(mean)
    assert mean == expected


# Test mean with non-overlapping windows
def test_mean_long_non_overlapping_window(long_data):
    ds = DataStream(long_data)
    mean = ds.mean(method="non-overlapping", window_size=2)
    expected = {"A": {"mean": 3.0}, "B": {"mean": 3.0}}
    assert mean == expected


# Test mean uncertainty
# =============================================================================


# Test mean uncertainty with simple data set
def test_mean_uncertainty_simple(simple_data):
    ds = DataStream(simple_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    expected = {"A": {"mean uncertainty": 0.5}}
    assert mean_uncertainty == expected


# Test mean uncertainty with long data set
def test_mean_uncertainty_long(long_data):
    ds = DataStream(long_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    expected = {
        "A": {"mean uncertainty": 0.6454972243679028},
        "B": {"mean uncertainty": 0.6454972243679028},
    }
    assert mean_uncertainty == expected


# Test Confidence Interval
# =============================================================================


def test_confidence_interval_simple(simple_data):
    ds = DataStream(simple_data)
    ci_df = ds.confidence_interval(window_size=1)
    expected = {"A": {"confidence interval": (0.8683934723883333, 3.131606527611667)}}
    assert ci_df == expected


def test_confidence_interval_long(long_data):
    ds = DataStream(long_data)
    ci_df = ds.confidence_interval(window_size=2)
    expected = {
        "A": {"confidence interval": (1.7348254402389105, 4.265174559761089)},
        "B": {"confidence interval": (1.7348254402389105, 4.265174559761089)},
    }
    assert ci_df == expected


# Test Trim
# =============================================================================


def test_trim_std(trim_data):
    ds = DataStream(trim_data)
    trim = ds.trim(
        column_name="A", window_size=1, method="std", start_time=3.0, threshold=4
    )
    expected = {"time": [3, 4, 5, 6, 7, 8, 9], "A": [4, 5, 6, 7, 8, 9, 10]}
    expected_df = pd.DataFrame(expected)
    pd.testing.assert_frame_equal(trim.df, expected_df)


def test_trim_threshold(trim_data):
    trim_data = trim_data.astype(float)
    ds = DataStream(trim_data)
    trim = ds.trim(
        column_name="A", window_size=1, method="threshold", start_time=3.0, threshold=4
    )
    assert trim is None


def test_trim_rolling_variance(trim_data):
    ds = DataStream(trim_data)
    trim = ds.trim(
        column_name="A",
        window_size=1,
        method="rolling_variance",
        start_time=3.0,
        threshold=4,
    )
    assert trim is None


def test_trim_invalid_method(trim_data):
    ds = DataStream(trim_data)
    with pytest.raises(ValueError):
        ds.trim(column_name="A", method="invalid_method")


# Test Compute Statistics
# =============================================================================


def test_compute_stats_simple(simple_data):
    ds = DataStream(simple_data)
    compute_stats = ds.compute_statistics(column_name="A", window_size=1)
    print(compute_stats)
    expected = {
        "A": {
            "mean": 2.0,
            "mean_uncertainty": 0.5773502691896258,
            "confidence_interval": (0.8683934723883333, 3.131606527611667),
            "pm_std": (1.4226497308103743, 2.5773502691896257),
        }
    }
    assert compute_stats == expected


def test_compute_stats_long(long_data):
    ds = DataStream(long_data)
    compute_stats = ds.compute_statistics(column_name="A", window_size=1)
    print(compute_stats)
    expected = {
        "A": {
            "mean": 3.0,
            "mean_uncertainty": 0.7071067811865476,
            "confidence_interval": (1.6140707088743669, 4.385929291125633),
            "pm_std": (2.2928932188134525, 3.7071067811865475),
        }
    }
    assert compute_stats == expected


# Test Optimal Window Size
# =============================================================================


def test_optimal_window_size_simple(simple_data):
    ds = DataStream(simple_data)
    opt_window_size = ds.optimal_window_size(method="sliding")
    print(opt_window_size)
    expected = {
        "A": {
            "optimal_window_size": 1,
            "min_std": 0.5773502691896258,
            "mean": 2.0,
            "ci": (0.8683934723883333, 3.131606527611667),
        }
    }
    assert opt_window_size == expected


def test_optimal_window_size_long(long_data):
    ds = DataStream(long_data)
    opt_window_size = ds.optimal_window_size(method="sliding")
    print(opt_window_size)
    expected = {
        "A": {
            "optimal_window_size": 1,
            "min_std": 0.7071067811865476,
            "mean": 3.0,
            "ci": (1.6140707088743669, 4.385929291125633),
        },
        "B": {
            "optimal_window_size": 1,
            "min_std": 0.7071067811865476,
            "mean": 3.0,
            "ci": (1.6140707088743669, 4.385929291125633),
        },
    }
    assert opt_window_size == expected


def test_opt_window_size_invalid_method(simple_data):
    ds = DataStream(simple_data)
    with pytest.raises(ValueError):
        ds.trim(column_name="A", method="invalid_method")


# Test Cumulative Statistics
# =============================================================================


def test_cumulative_stats_simple(simple_data):
    ds = DataStream(simple_data)
    cumulative_stats = ds.cumulative_statistics(window_size=1)
    expected = {
        "A": {
            "cumulative_mean": [1.0, 1.5, 2.0],
            "cumulative_uncertainty": [np.nan, 0.7071067811865476, 1.0],
            "standard_error": [np.nan, 0.5, 0.5773502691896258],
        }
    }
    assert_equal(cumulative_stats, expected)


def test_cumulative_stats_long(long_data):
    ds = DataStream(long_data)
    cumulative_stats = ds.cumulative_statistics(window_size=1)
    expected = {
        "A": {
            "cumulative_mean": [1.0, 1.5, 2.0, 2.5, 3.0],
            "cumulative_uncertainty": [
                np.nan,
                0.7071067811865476,
                1.0,
                1.2909944487358056,
                1.5811388300841898,
            ],
            "standard_error": [
                np.nan,
                0.5,
                0.5773502691896258,
                0.6454972243679028,
                0.7071067811865476,
            ],
        },
        "B": {
            "cumulative_mean": [5.0, 4.5, 4.0, 3.5, 3.0],
            "cumulative_uncertainty": [
                np.nan,
                0.7071067811865476,
                1.0,
                1.2909944487358056,
                1.5811388300841898,
            ],
            "standard_error": [
                np.nan,
                0.5,
                0.5773502691896258,
                0.6454972243679028,
                0.7071067811865476,
            ],
        },
    }
    assert_equal(cumulative_stats, expected)


def test_cumulative_stats_empty(nan_data):
    ds = DataStream(nan_data)
    cumulative_stats = ds.cumulative_statistics(window_size=1)
    expected = {"A": {"error": "No data available for column 'A'"}}
    assert cumulative_stats == expected


# Test Additional Data
# =============================================================================


def test_additional_data_simple(simple_data):
    ds = DataStream(simple_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # Ignore UserWarnings
        additional_data = ds.additional_data(window_size=1, method="sliding")
    expected = {
        "A": {
            "A_est": 0.3910010411753345,
            "p_est": 0.8547556456757277,
            "n_current": 3,
            "current_sem": 0.1528818142001956,
            "target_sem": 0.13759363278017603,
            "n_target": 3.393548707049326,
            "additional_samples": 1,
        }
    }
    assert additional_data == expected


def test_additional_data_long(long_data):
    ds = DataStream(long_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # Ignore UserWarnings
        additional_data = ds.additional_data(window_size=1, method="sliding")
    print(additional_data)
    expected = {
        "A": {
            "A_est": 0.3803501348616604,
            "p_est": 0.8838111262612045,
            "n_current": 5,
            "" "current_sem": 0.09171198805673249,
            "target_sem": 0.08254078925105925,
            "n_target": 5.633041271578334,
            "additional_samples": 1,
        },
        "B": {
            "A_est": 0.3803501348616604,
            "p_est": 0.8838111262612045,
            "n_current": 5,
            "current_sem": 0.09171198805673249,
            "target_sem": 0.08254078925105925,
            "n_target": 5.633041271578334,
            "additional_samples": 1,
        },
    }
    assert additional_data == expected


def mock_cumulative_statistics_missing(col_name, method, window_size):
    return {
        "A": {
            "cumulative_uncertainty": [0.5, 0.4, 0.3],
        },
        "B": {
            # Column B intentionally missing cumulative_uncertainty for testing
        },
    }


def test_additional_data_missing_cumulative(long_data):
    ds = DataStream(long_data)
    ds.cumulative_statistics = mock_cumulative_statistics_missing
    additional_data = ds.additional_data(column_name="B", reduction_factor=0.1)
    print(additional_data)
    expected = {"B": {"error": "No cumulative SEM data for column 'B'"}}
    assert additional_data == expected


# Test Effective Sample Size Below
# =============================================================================


def test_effective_sample_size_below_simple(simple_data):
    ds = DataStream(simple_data)
    effective_sample_size_below = ds.effective_sample_size_below(column_names="A")
    print(effective_sample_size_below)
    expected = {"A": 3}
    assert effective_sample_size_below == expected


def test_effective_sample_size_below_long(long_data):
    ds = DataStream(long_data)
    effective_sample_size_below = ds.effective_sample_size_below(column_names="A")
    print(effective_sample_size_below)
    expected = {"A": 5}
    assert effective_sample_size_below == expected
