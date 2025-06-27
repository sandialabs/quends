import warnings

import numpy as np
import pandas as pd
import pytest

from quends import DataStream


# === Fixtures ===
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
    return pd.DataFrame({"time": list(range(10)), "A": list(range(1, 11))})


@pytest.fixture
def nan_data():
    return pd.DataFrame({"A": [None, None, None]})


@pytest.fixture
def no_valid_data():
    return pd.DataFrame({"time": [0, 1], "A": [1, 2]})


# === Initialization ===
def test_init_simple(simple_data):
    ds = DataStream(simple_data)
    assert len(ds) == 3
    assert ds.variables().tolist() == ["A"]


def test_init_empty(empty_data):
    ds = DataStream(empty_data)
    assert len(ds) == 0
    assert ds.variables().tolist() == []


# === Mean ===
def test_mean_simple(simple_data):
    ds = DataStream(simple_data)
    assert ds.mean(window_size=1) == {"A": 2.0}


def test_mean_empty(empty_data):
    ds = DataStream(empty_data)
    assert ds.mean() == {}


def test_mean_long(long_data):
    ds = DataStream(long_data)
    assert ds.mean() == {"A": 3.0, "B": 3.0}


def test_mean_long_overlapping_window(long_data):
    ds = DataStream(long_data)
    assert ds.mean() == {"A": 3.0, "B": 3.0}


def test_mean_long_non_overlapping_window(long_data):
    ds = DataStream(long_data)
    assert ds.mean(method="non-overlapping", window_size=2) == {"A": 2.5, "B": 3.5}


# === Mean Uncertainty ===
def test_mean_uncertainty_simple(simple_data):
    ds = DataStream(simple_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    assert np.isnan(mean_uncertainty["A"])


def test_mean_uncertainty_long(long_data):
    ds = DataStream(long_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    assert mean_uncertainty == {"A": 1.0, "B": 1.0}


# === Confidence Interval ===
def test_confidence_interval_simple(simple_data):
    ds = DataStream(simple_data)
    expected = {"A": (0.8683934723883333, 3.131606527611667)}
    assert ds.confidence_interval(window_size=1) == expected


def test_confidence_interval_long(long_data):
    ds = DataStream(long_data)
    expected = {"A": (0.54, 4.46), "B": (1.54, 5.46)}
    assert ds.confidence_interval(window_size=2) == expected


# === Trim ===
def test_trim_std(trim_data):
    ds = DataStream(trim_data)
    expected = {
        "results": None,
        "metadata": [
            {"operation": "is_stationary", "options": {"columns": "A"}},
            {
                "operation": "trim",
                "options": {
                    "column_name": "A",
                    "batch_size": 1,
                    "start_time": 3.0,
                    "method": "std",
                    "threshold": 4,
                    "robust": True,
                    "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
                },
            },
        ],
        "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
    }
    assert (
        ds.trim(
            column_name="A", batch_size=1, method="std", start_time=3.0, threshold=4
        )
        == expected
    )


def test_trim_threshold(trim_data):
    ds = DataStream(trim_data.astype(float))
    expected = {
        "results": None,
        "metadata": [
            {"operation": "is_stationary", "options": {"columns": "A"}},
            {
                "operation": "trim",
                "options": {
                    "column_name": "A",
                    "batch_size": 1,
                    "start_time": 3.0,
                    "method": "threshold",
                    "threshold": 4,
                    "robust": True,
                    "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
                },
            },
        ],
        "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
    }
    assert (
        ds.trim(
            column_name="A",
            batch_size=1,
            method="threshold",
            start_time=3.0,
            threshold=4,
        )
        == expected
    )


def test_trim_rolling_variance(trim_data):
    ds = DataStream(trim_data)
    expected = {
        "results": None,
        "metadata": [
            {"operation": "is_stationary", "options": {"columns": "A"}},
            {
                "operation": "trim",
                "options": {
                    "column_name": "A",
                    "batch_size": 1,
                    "start_time": 3.0,
                    "method": "rolling_variance",
                    "threshold": 4,
                    "robust": True,
                    "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
                },
            },
        ],
        "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
    }
    assert (
        ds.trim(
            column_name="A",
            batch_size=1,
            method="rolling_variance",
            start_time=3.0,
            threshold=4,
        )
        == expected
    )


def test_trim_invalid_method(trim_data):
    ds = DataStream(trim_data)
    result = ds.trim(column_name="A", method="invalid_method")
    expected = {
        "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
        "metadata": [
            {"operation": "is_stationary", "options": {"columns": "A"}},
            {
                "operation": "trim",
                "options": {
                    "batch_size": 10,
                    "column_name": "A",
                    "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
                    "method": "invalid_method",
                    "robust": True,
                    "start_time": 0.0,
                    "threshold": None,
                },
            },
        ],
        "results": None,
    }
    assert result == expected


def test_trim_missing_threshold(long_data):
    ds = DataStream(long_data)
    result = ds.trim(column_name="A", method="threshold")
    expected = {
        "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
        "metadata": [
            {"operation": "is_stationary", "options": {"columns": "A"}},
            {
                "operation": "trim",
                "options": {
                    "batch_size": 10,
                    "column_name": "A",
                    "message": "Column 'A' is not stationary. Steady-state trimming requires stationary data.",
                    "method": "threshold",
                    "robust": True,
                    "start_time": 0.0,
                    "threshold": None,
                },
            },
        ],
        "results": None,
    }
    assert result == expected


# === Compute Statistics ===
def test_compute_stats_simple(simple_data):
    ds = DataStream(simple_data)
    expected = {
        "A": {
            "mean": 2.0,
            "mean_uncertainty": 0.5773502691896258,
            "confidence_interval": (0.8683934723883333, 3.131606527611667),
            "pm_std": (1.4226497308103743, 2.5773502691896257),
            "effective_sample_size": 3,
            "window_size": 1,
        },
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": "A", "alpha": 0.05},
            },
            {
                "operation": "compute_statistics",
                "options": {
                    "column_name": "A",
                    "ddof": 1,
                    "method": "non-overlapping",
                    "window_size": 1,
                },
            },
        ],
    }
    assert ds.compute_statistics(column_name="A", window_size=1) == expected


def test_compute_stats_long(long_data):
    ds = DataStream(long_data)
    expected = {
        "A": {
            "mean": 3.0,
            "mean_uncertainty": 0.7071067811865476,
            "confidence_interval": (1.6140707088743669, 4.385929291125633),
            "pm_std": (2.2928932188134525, 3.7071067811865475),
            "effective_sample_size": 5,
            "window_size": 1,
        },
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": "A", "alpha": 0.05},
            },
            {
                "operation": "compute_statistics",
                "options": {
                    "column_name": "A",
                    "ddof": 1,
                    "method": "non-overlapping",
                    "window_size": 1,
                },
            },
        ],
    }
    assert ds.compute_statistics(column_name="A", window_size=1) == expected


def test_compute_stats_ci_not_computed(long_data):
    ds = DataStream(long_data)
    original_ci_method = ds.confidence_interval
    ds.confidence_interval = lambda *a, **k: {"A": None}
    result = ds.compute_statistics(column_name="A", window_size=1)
    ds.confidence_interval = original_ci_method
    assert "A" in result


# === Optimal Window Size ===
def test_optimal_window_size_simple(simple_data):
    ds = DataStream(simple_data)
    assert ds.optimal_window_size() == 1


def test_optimal_window_size_long(long_data):
    ds = DataStream(long_data)
    assert ds.optimal_window_size() == 1


# === Cumulative Statistics ===
def test_cumulative_stats_simple(simple_data):
    ds = DataStream(simple_data)
    result = ds.cumulative_statistics(window_size=1)
    expected = {
        "A": {
            "cumulative_mean": [1.0, 1.5, 2.0],
            "cumulative_uncertainty": [np.nan, 0.7071067811865476, 1.0],
            "standard_error": [np.nan, 0.5, 0.5773502691896258],
            "window_size": 1,
        },
        "metadata": [
            {
                "operation": "cumulative_statistics",
                "options": {
                    "column_name": None,
                    "method": "non-overlapping",
                    "window_size": 1,
                },
            }
        ],
    }
    for key in expected["A"]:
        if isinstance(expected["A"][key], list):
            np.testing.assert_equal(result["A"][key], expected["A"][key])


def test_cumulative_stats_long(long_data):
    ds = DataStream(long_data)
    result = ds.cumulative_statistics(window_size=1)
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
            "window_size": 1,
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
            "window_size": 1,
        },
        "metadata": [
            {
                "operation": "cumulative_statistics",
                "options": {
                    "column_name": None,
                    "method": "non-overlapping",
                    "window_size": 1,
                },
            }
        ],
    }
    for col in ["A", "B"]:
        for key in expected[col]:
            np.testing.assert_equal(result[col][key], expected[col][key])


def test_cumulative_stats_empty(nan_data):
    ds = DataStream(nan_data)
    expected = {
        "A": {"error": "No data available for column 'A'"},
        "metadata": [
            {
                "operation": "cumulative_statistics",
                "options": {
                    "column_name": None,
                    "method": "non-overlapping",
                    "window_size": 1,
                },
            }
        ],
    }
    assert ds.cumulative_statistics(window_size=1) == expected


# === Additional Data ===
import pytest


def assert_nested_approx(a, b, rel=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_approx(a[k], b[k], rel=rel)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for i, j in zip(a, b):
            assert_nested_approx(i, j, rel=rel)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=rel)
    else:
        assert a == b


def test_additional_data_simple(simple_data):
    ds = DataStream(simple_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    expected = {
        "A": {
            "A_est": 0.3910010411753345,
            "p_est": 0.8547556456757277,
            "n_current": 3,
            "current_sem": 0.1528818142001956,
            "target_sem": 0.13759363278017603,
            "n_target": 3.393548707049326,
            "additional_samples": 1,
            "window_size": 1,
        },
        "metadata": [
            {
                "operation": "additional_data",
                "options": {
                    "column_name": None,
                    "ddof": 1,
                    "method": "sliding",
                    "window_size": 1,
                    "reduction_factor": 0.1,
                },
            }
        ],
    }
    assert_nested_approx(result, expected)


def test_additional_data_long(long_data):
    ds = DataStream(long_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    expected = {
        "A": {
            "A_est": 0.3803501348616604,
            "p_est": 0.8838111262612045,
            "n_current": 5,
            "current_sem": 0.09171198805673249,
            "target_sem": 0.08254078925105925,
            "n_target": 5.633041271578334,
            "additional_samples": 1,
            "window_size": 1,
        },
        "B": {
            "A_est": 0.3803501348616604,
            "p_est": 0.8838111262612045,
            "n_current": 5,
            "current_sem": 0.09171198805673249,
            "target_sem": 0.08254078925105925,
            "n_target": 5.633041271578334,
            "additional_samples": 1,
            "window_size": 1,
        },
        "metadata": [
            {
                "operation": "additional_data",
                "options": {
                    "column_name": None,
                    "ddof": 1,
                    "method": "sliding",
                    "window_size": 1,
                    "reduction_factor": 0.1,
                },
            }
        ],
    }
    assert_nested_approx(result, expected)


def mock_cumulative_statistics_missing(col_name, method, window_size):
    return {"A": {"cumulative_uncertainty": [0.5, 0.4, 0.3]}, "B": {}}


def test_additional_data_missing_cumulative(long_data):
    ds = DataStream(long_data)
    ds.cumulative_statistics = mock_cumulative_statistics_missing
    additional_data = ds.additional_data(column_name="B", reduction_factor=0.1)
    expected = {
        "B": {"error": "No cumulative SEM data for column 'B'"},
        "metadata": [
            {
                "operation": "additional_data",
                "options": {
                    "column_name": "B",
                    "ddof": 1,
                    "method": "sliding",
                    "reduction_factor": 0.1,
                    "window_size": None,
                },
            }
        ],
    }
    assert additional_data == expected


# === Effective Sample Size Below ===
def test_effective_sample_size_below_simple(simple_data):
    ds = DataStream(simple_data)
    assert ds.effective_sample_size_below(column_names="A") == {"A": 0}


def test_effective_sample_size_below_long(long_data):
    ds = DataStream(long_data)
    assert ds.effective_sample_size_below(column_names="A") == {"A": 0}


def test_effective_sample_size_below_invalid_column(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size_below(column_names="C")
    assert result == {"C": 0}


def test_effective_sample_size_below_empty_column():
    empty_data = {
        "time": [0, 1, 2, 3, 4],
        "A": [None, None, None, None, None],
        "B": [5, 4, 3, 2, 1],
    }
    ds = DataStream(pd.DataFrame(empty_data))
    result = ds.effective_sample_size_below(column_names="A")
    assert result == {"A": 0}


# === Stationary ===
def test_is_stationary(stationary_data):
    ds = DataStream(stationary_data)
    assert ds.is_stationary(columns="A") == {"A": "Error: Invalid input, x is constant"}


def test_is_not_stationary(long_data):
    ds = DataStream(long_data)
    out = ds.is_stationary(columns="A")
    if hasattr(np, "False_"):
        assert out == {"A": np.False_}
    else:
        assert out == {"A": False}


# === Head ===
def test_head(long_data):
    ds = DataStream(long_data)
    expected = pd.DataFrame(
        {"time": [0, 1, 2, 3, 4], "A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}
    )
    pd.testing.assert_frame_equal(ds.head(5), expected)


# === Process Column Error ===
def test_process_column_missing_method(simple_data):
    ds = DataStream(simple_data)
    with pytest.raises(ValueError):
        ds._process_column(column_data="A", estimated_window=1, method="invalid_method")


# === Find Steady State Std ===
def test_find_steady_state_std(trim_data):
    ds = DataStream(trim_data)
    assert ds.find_steady_state_std(data=ds.df, column_name="A", window_size=1) == 0


def test_find_steady_state_std_non_robust(trim_data):
    ds = DataStream(trim_data)
    assert (
        ds.find_steady_state_std(
            data=ds.df, column_name="A", window_size=2, robust=False
        )
        == 3
    )


def test_find_steady_state_not_valid(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_std(
        data=ds.df, column_name=["time", "A"], window_size=1
    )
    assert result is None


# === Find Steady State Threshold ===
def test_find_steady_state_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=2, threshold=0.1
    )
    assert result == 1


def test_find_steady_state_long_data(long_data):
    ds = DataStream(long_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=2, threshold=0.1
    )
    assert result is None


def test_find_steady_state_trim_data(trim_data):
    ds = DataStream(trim_data)
    result = ds.find_steady_state_threshold(
        data=ds.df, column_name="A", window_size=3, threshold=0.5
    )
    assert result == 2


def test_find_steady_state_with_start_time(long_data):
    ds = DataStream(long_data)
    pass  #


# === Find Steady State Rolling Variance ===
def test_find_steady_state_rolling_variance_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.find_steady_state_rolling_variance(
        data=ds.df, column_name="A", window_size=3
    )
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


# === effective_sample_size ===
def test_effective_sample_size_empty(empty_data):
    ds = DataStream(empty_data)
    expected = {
        "results": {},
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"alpha": 0.05, "column_names": None},
            }
        ],
    }
    assert ds.effective_sample_size() == expected


def test_effective_sample_size_nan(nan_data):
    ds = DataStream(nan_data)
    result = ds.effective_sample_size(column_names=["A"])
    expected = {
        "results": {
            "A": {
                "effective_sample_size": None,
                "message": "No data available for computation.",
            }
        },
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": ["A"], "alpha": 0.05},
            }
        ],
    }
    assert result == expected


def test_effective_sample_size_simple(simple_data):
    ds = DataStream(simple_data)
    result = ds.effective_sample_size(column_names=["A"])
    expected = {
        "results": {"A": 3},
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": ["A"], "alpha": 0.05},
            }
        ],
    }
    assert result == expected


def test_effective_sample_size_long_data(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size(column_names=["A", "B"])
    expected = {
        "results": {"A": 5, "B": 5},
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": ["A", "B"], "alpha": 0.05},
            }
        ],
    }
    assert result == expected


def test_effective_sample_size_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.effective_sample_size(column_names=["A"])
    expected = {
        "results": {"A": 5},
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": ["A"], "alpha": 0.05},
            }
        ],
    }
    assert result == expected


def test_effective_sample_size_trim_data(trim_data):
    ds = DataStream(trim_data)
    result = ds.effective_sample_size(column_names=["A"])
    expected = {
        "results": {"A": 5},
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": ["A"], "alpha": 0.05},
            }
        ],
    }
    assert result == expected


def test_effective_sample_size_missing_col(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size(column_names=["C"])
    expected = {
        "results": {"C": {"message": "Column 'C' not found in the DataStream."}},
        "metadata": [
            {
                "operation": "effective_sample_size",
                "options": {"column_names": ["C"], "alpha": 0.05},
            }
        ],
    }
    assert result == expected
