# Import statements
import pandas as pd
import pytest

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
