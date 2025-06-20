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
            "time": [0, 1, 2, 3, 4],
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
        }
    )


@pytest.fixture
def stationary_data():
    return pd.DataFrame(
        {"time": [0, 1, 2, 3, 4], "A": [1, 1, 1, 1, 1], "B": [2, 2, 2, 2, 2]}
    )


@pytest.fixture
def trim_data():
    return pd.DataFrame(
        {"time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    )


@pytest.fixture
def nan_data():
    return pd.DataFrame({"A": [None, None, None]})


@pytest.fixture
def no_valid_data():
    return pd.DataFrame({"time": [0, 1], "A": [1, 2]})


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


def test_trim_missing_threshold(long_data):
    ds = DataStream(long_data)
    with pytest.raises(
        ValueError, match="Threshold must be specified for the 'threshold' method."
    ):
        ds.trim(column_name="A", method="threshold")


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


def test_compute_stats_ci_not_computed(long_data):
    ds = DataStream(long_data)

    # Mocking the confidence_interval method to return None for column 'A'
    original_ci_method = ds.confidence_interval
    ds.confidence_interval = lambda column_name, ddof, method, window_size: {
        "A": {"confidence interval": None}
    }

    result = ds.compute_statistics(column_name="A")
    print(result)

    # Restore the original method
    ds.confidence_interval = original_ci_method

    assert result["A"]["error"] == "Confidence interval not computed for column 'A'"


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


def test_effective_sample_size_below_invalid_column(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size_below(column_names="C")
    assert result["C"]["message"] == "Column 'C' not found in the DataStream."


def test_effective_sample_size_below_empty_column():
    empty_data = {
        "time": [0, 1, 2, 3, 4],
        "A": [None, None, None, None, None],  # Column A has no data
        "B": [5, 4, 3, 2, 1],
    }
    ds = DataStream(pd.DataFrame(empty_data))

    # Call the method with column 'A' which is empty
    result = ds.effective_sample_size_below(column_names="A")

    # Check the result for the expected message
    assert result["A"]["effective_sample_size"] is None
    assert result["A"]["message"] == "No data available for computation."


# Test Stationary
# =============================================================================


def test_is_stationary(stationary_data):
    ds = DataStream(stationary_data)
    stationary = ds.is_stationary(columns="A")
    print(stationary)
    expected = {"A": "Error: Invalid input, x is constant"}
    assert stationary == expected


def test_is_not_stationary(long_data):
    ds = DataStream(long_data)
    with pytest.warns(Warning):
        not_stationary = ds.is_stationary(columns="A")
    expected = {"A": False}
    print(not_stationary)
    assert not_stationary == expected


# Test Head
# =============================================================================


def test_head(long_data):
    ds = DataStream(long_data)
    result = ds.head(5)
    expected = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


# Test Process Column
# =============================================================================


def test_process_column_missing_method(simple_data):
    ds = DataStream(simple_data)
    with pytest.raises(
        ValueError, match="Invalid method. Choose 'sliding' or 'non-overlapping'."
    ):
        ds._process_column(column_data="A", estimated_window=1, method="invalid_method")


# Test Find Steady State Std
# =============================================================================


def test_find_steady_state_std(trim_data):
    ds = DataStream(trim_data)
    result = ds.find_steady_state_std(data=ds.df, column_name="A", window_size=1)
    expected = 0
    assert result == expected


def test_find_steady_state_std_non_robust(trim_data):
    ds = DataStream(trim_data)
    result = ds.find_steady_state_std(
        data=ds.df, column_name="A", window_size=2, robust=False
    )
    expected = 3
    assert result == expected


def test_find_steady_state_not_valid(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_std(
        data=ds.df, column_name=["time", "A"], window_size=1
    )
    assert result is None


# Test Find Steady State Threshold
# =============================================================================


def test_find_steady_state_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=2, threshold=0.1
    )
    assert result == 2


def test_find_steady_state_long_data(long_data):
    ds = DataStream(long_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=2, threshold=0.1
    )
    assert result == 2


def test_find_steady_state_trim_data(trim_data):
    ds = DataStream(trim_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=3, threshold=0.5
    )
    assert result == 4


def test_find_steady_state_no_valid_data(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=2, threshold=0.5
    )
    assert result is None


def test_find_steady_state_with_start_time(long_data):
    ds = DataStream(long_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=2, threshold=0.1, start_time=1
    )
    assert result == 3


# Test Find Steady State Rolling Variance
# =============================================================================


def test_find_steady_state_rolling_variance_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.find_steady_state_rolling_variance(
        data=ds.df, column_name="A", window_size=3
    )
    print(result)
    assert result is None


def test_find_steady_state_none_rolling_variance(long_data):
    ds = DataStream(long_data)
    result = ds.find_steady_state_rolling_variance(
        data=long_data, column_name="A", window_size=3, threshold=0.1
    )
    assert result is None


def test_find_steady_state_rolling_variance_not_valid(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_rolling_variance(
        data=ds.df, column_name="A", window_size=1
    )
    assert result is None


# Test effective_sample_size
# =============================================================================


def test_effective_sample_size_empty(empty_data):
    ds = DataStream(empty_data)
    result = ds.effective_sample_size()
    assert result == {}


def test_effective_sample_size_nan(nan_data):
    ds = DataStream(nan_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert result["A"]["effective_sample_size"] is None
    assert result["A"]["message"] == "No data available for computation."


def test_effective_sample_size_simple(simple_data):
    ds = DataStream(simple_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert "A" in result
    assert result["A"] is not None


def test_effective_sample_size_long_data(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size(column_names=["A", "B"])
    assert "A" in result
    assert "B" in result
    assert result["A"] is not None
    assert result["B"] is not None


def test_effective_sample_size_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert "A" in result
    assert result["A"] is not None


def test_effective_sample_size_trim_data(trim_data):
    ds = DataStream(trim_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert "A" in result
    assert result["A"] is not None


def test_effective_sample_size_missing_col(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size(column_names=["C"])
    assert result["C"]["message"] == "Column 'C' not found in the DataStream."
