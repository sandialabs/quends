import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
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
    return pd.DataFrame({
        "time": [0, 1, 2, 3, 4],
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
    })

@pytest.fixture
def stationary_data():
    return pd.DataFrame({"time": [0, 1, 2, 3, 4], "A": [1, 1, 1, 1, 1], "B": [2, 2, 2, 2, 2]})

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
    assert ds.variables().tolist() == ['A']

def test_init_empty(empty_data):
    ds = DataStream(empty_data)
    assert len(ds) == 0
    assert ds.variables().tolist() == []

# === Mean ===
def test_mean_simple(simple_data):
    ds = DataStream(simple_data)
    assert ds.mean(window_size=1) == {'A': 2.0}

def test_mean_empty(empty_data):
    ds = DataStream(empty_data)
    assert ds.mean() == {}

def test_mean_long(long_data):
    ds = DataStream(long_data)
    assert ds.mean() == {'A': 3.0, 'B': 3.0}

def test_mean_long_overlapping_window(long_data):
    ds = DataStream(long_data)
    assert ds.mean() == {'A': 3.0, 'B': 3.0}

def test_mean_long_non_overlapping_window(long_data):
    ds = DataStream(long_data)
    assert ds.mean(method="non-overlapping", window_size=2) == {'A': 2.5, 'B': 3.5}

# === Mean Uncertainty ===
def test_mean_uncertainty_simple(simple_data):
    ds = DataStream(simple_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    assert np.isnan(mean_uncertainty['A'])

def test_mean_uncertainty_long(long_data):
    ds = DataStream(long_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    assert mean_uncertainty == {'A': 1.0, 'B': 1.0}

# === Confidence Interval ===
def test_confidence_interval_simple(simple_data):
    ds = DataStream(simple_data)
    expected = {"A": (0.8683934723883333, 3.131606527611667)}
    assert ds.confidence_interval(window_size=1) == expected

def test_confidence_interval_long(long_data):
    ds = DataStream(long_data)
    expected = {
        "A": (1.7348254402389105, 4.265174559761089),
        "B": (1.7348254402389105, 4.265174559761089)
    }
    assert ds.confidence_interval(window_size=2) == expected

# === Trim ===
def test_trim_std(trim_data):
    ds = DataStream(trim_data)
    expected = {
        'results': None,
        'metadata': [
            {'operation': 'is_stationary', 'options': {'columns': 'A'}},
            {'operation': 'trim', 'options': {
                'column_name': 'A', 'batch_size': 1, 'start_time': 3.0,
                'method': 'std', 'threshold': 4, 'robust': True,
                'message': "Column 'A' is not stationary. Steady-state trimming requires stationary data."
            }}
        ],
        'message': "Column 'A' is not stationary. Steady-state trimming requires stationary data."
    }
    assert ds.trim(column_name="A", batch_size=1, method="std", start_time=3.0, threshold=4) == expected

def test_trim_threshold(trim_data):
    ds = DataStream(trim_data.astype(float))
    expected = {
        'results': None,
        'metadata': [
            {'operation': 'is_stationary', 'options': {'columns': 'A'}},
            {'operation': 'trim', 'options': {
                'column_name': 'A', 'batch_size': 1, 'start_time': 3.0,
                'method': 'threshold', 'threshold': 4, 'robust': True,
                'message': "Column 'A' is not stationary. Steady-state trimming requires stationary data."
            }}
        ],
        'message': "Column 'A' is not stationary. Steady-state trimming requires stationary data."
    }
    assert ds.trim(column_name="A", batch_size=1, method="threshold", start_time=3.0, threshold=4) == expected

def test_trim_rolling_variance(trim_data):
    ds = DataStream(trim_data)
    expected = {
        'results': None,
        'metadata': [
            {'operation': 'is_stationary', 'options': {'columns': 'A'}},
            {'operation': 'trim', 'options': {
                'column_name': 'A', 'batch_size': 1, 'start_time': 3.0,
                'method': 'rolling_variance', 'threshold': 4, 'robust': True,
                'message': "Column 'A' is not stationary. Steady-state trimming requires stationary data."
            }}
        ],
        'message': "Column 'A' is not stationary. Steady-state trimming requires stationary data."
    }
    assert ds.trim(column_name="A", batch_size=1, method="rolling_variance", start_time=3.0, threshold=4) == expected

def test_trim_invalid_method(trim_data):
    ds = DataStream(trim_data)
    with pytest.raises(ValueError):
        ds.trim(column_name="A", method="invalid_method")

def test_trim_missing_threshold(long_data):
    ds = DataStream(long_data)
    with pytest.raises(Exception):
        ds.trim(column_name="A", method="threshold")


# === Compute Statistics ===
def test_compute_stats_simple(simple_data):
    ds = DataStream(simple_data)
    expected = {'A': {'mean': 2.0, 'mean_uncertainty': 0.5773502691896258,
                      'confidence_interval': (0.8683934723883333, 3.131606527611667),
                      'pm_std': (1.4226497308103743, 2.5773502691896257),
                      'effective_sample_size': 3, 'window_size': 1},
                'metadata': [
                    {'operation': 'effective_sample_size', 'options': {'column_names': 'A', 'alpha': 0.05}},
                    {'operation': 'compute_statistics', 'options': {'column_name': 'A', 'ddof': 1, 'method': 'non-overlapping', 'window_size': 1}}
                ]}
    assert ds.compute_statistics(column_name="A", window_size=1) == expected

def test_compute_stats_long(long_data):
    ds = DataStream(long_data)
    expected = {'A': {'mean': 3.0, 'mean_uncertainty': 0.7071067811865476,
                      'confidence_interval': (1.6140707088743669, 4.385929291125633),
                      'pm_std': (2.2928932188134525, 3.7071067811865475),
                      'effective_sample_size': 5, 'window_size': 1},
                'metadata': [
                    {'operation': 'effective_sample_size', 'options': {'column_names': 'A', 'alpha': 0.05}},
                    {'operation': 'compute_statistics', 'options': {'column_name': 'A', 'ddof': 1, 'method': 'non-overlapping', 'window_size': 1}}
                ]}
    assert ds.compute_statistics(column_name="A", window_size=1) == expected

def test_compute_stats_ci_not_computed(long_data):
    ds = DataStream(long_data)
    original_ci_method = ds.confidence_interval
    ds.confidence_interval = lambda *a, **k: {"A": None}
    result = ds.compute_statistics(column_name="A", window_size=1)
    ds.confidence_interval = original_ci_method
    assert 'A' in result

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
    expected = {'A': {'cumulative_mean': [1.0, 1.5, 2.0],
                      'cumulative_uncertainty': [np.nan, 0.7071067811865476, 1.0],
                      'standard_error': [np.nan, 0.5, 0.5773502691896258], 'window_size': 1},
                'metadata': [{'operation': 'cumulative_statistics', 'options': {'column_name': None, 'method': 'non-overlapping', 'window_size': 1}}]}
    for key in expected['A']:
        if isinstance(expected['A'][key], list):
            np.testing.assert_equal(result['A'][key], expected['A'][key])

def test_cumulative_stats_long(long_data):
    ds = DataStream(long_data)
    result = ds.cumulative_statistics(window_size=1)
    expected = {'A': {'cumulative_mean': [1.0, 1.5, 2.0, 2.5, 3.0],
                      'cumulative_uncertainty': [np.nan, 0.7071067811865476, 1.0, 1.2909944487358056, 1.5811388300841898],
                      'standard_error': [np.nan, 0.5, 0.5773502691896258, 0.6454972243679028, 0.7071067811865476], 'window_size': 1},
                'B': {'cumulative_mean': [5.0, 4.5, 4.0, 3.5, 3.0],
                      'cumulative_uncertainty': [np.nan, 0.7071067811865476, 1.0, 1.2909944487358056, 1.5811388300841898],
                      'standard_error': [np.nan, 0.5, 0.5773502691896258, 0.6454972243679028, 0.7071067811865476], 'window_size': 1},
                'metadata': [{'operation': 'cumulative_statistics', 'options': {'column_name': None, 'method': 'non-overlapping', 'window_size': 1}}]}
    for col in ['A', 'B']:
        for key in expected[col]:
            np.testing.assert_equal(result[col][key], expected[col][key])

def test_cumulative_stats_empty(nan_data):
    ds = DataStream(nan_data)
    expected = {'A': {'error': "No data available for column 'A'"},
                'metadata': [{'operation': 'cumulative_statistics', 'options': {'column_name': None, 'method': 'non-overlapping', 'window_size': 1}}]}
    assert ds.cumulative_statistics(window_size=1) == expected

# === Additional Data ===
def test_additional_data_simple(simple_data):
    ds = DataStream(simple_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    expected = {'A': {'A_est': 0.3910010411753347, 'p_est': 0.8547556456757269, 'n_current': 3,
                      'current_sem': 0.15288181420019578, 'target_sem': 0.1375936327801762,
                      'n_target': 3.393548707049327, 'additional_samples': 1, 'window_size': 1},
                'metadata': [{'operation': 'additional_data', 'options': {'column_name': None, 'ddof': 1, 'method': 'sliding', 'window_size': 1, 'reduction_factor': 0.1}}]}
    assert result == expected

def test_additional_data_long(long_data):
    ds = DataStream(long_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    expected = {'A': {'A_est': 0.38035013491470165, 'p_est': 0.883811126151829, 'n_current': 5,
                      'current_sem': 0.0917119880856664, 'target_sem': 0.08254078927709976,
                      'n_target': 5.633041271661439, 'additional_samples': 1, 'window_size': 1},
                'B': {'A_est': 0.38035013491470165, 'p_est': 0.883811126151829, 'n_current': 5,
                      'current_sem': 0.0917119880856664, 'target_sem': 0.08254078927709976,
                      'n_target': 5.633041271661439, 'additional_samples': 1, 'window_size': 1},
                'metadata': [{'operation': 'additional_data', 'options': {'column_name': None, 'ddof': 1, 'method': 'sliding', 'window_size': 1, 'reduction_factor': 0.1}}]}
    assert result == expected

def mock_cumulative_statistics_missing(col_name, method, window_size):
    return {
        "A": {"cumulative_uncertainty": [0.5, 0.4, 0.3]},
        "B": {}
    }

def test_additional_data_missing_cumulative(long_data):
    ds = DataStream(long_data)
    ds.cumulative_statistics = mock_cumulative_statistics_missing
    additional_data = ds.additional_data(column_name="B", reduction_factor=0.1)
    expected = {"B": {"error": "No cumulative SEM data for column 'B'"}}
    assert additional_data == expected

# === Effective Sample Size Below ===
def test_effective_sample_size_below_simple(simple_data):
    ds = DataStream(simple_data)
    assert ds.effective_sample_size_below(column_names="A") == {'A': 0}

def test_effective_sample_size_below_long(long_data):
    ds = DataStream(long_data)
    assert ds.effective_sample_size_below(column_names="A") == {'A': 0}

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
    result = ds.effective_sample_size_below(column_names="A")
    assert result["A"]["effective_sample_size"] is None
    assert result["A"]["message"] == "No data available for computation."

# === Stationary ===
def test_is_stationary(stationary_data):
    ds = DataStream(stationary_data)
    assert ds.is_stationary(columns="A") == {'A': 'Error: Invalid input, x is constant'}

def test_is_not_stationary(long_data):
    ds = DataStream(long_data)
    out = ds.is_stationary(columns="A")
    if hasattr(np, "False_"):
        assert out == {'A': np.False_}
    else:
        assert out == {'A': False}

# === Head ===
def test_head(long_data):
    ds = DataStream(long_data)
    expected = pd.DataFrame({"time": [0,1,2,3,4], "A": [1,2,3,4,5], "B": [5,4,3,2,1]})
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
    assert ds.find_steady_state_std(data=ds.df, column_name="A", window_size=2, robust=False) == 3

def test_find_steady_state_not_valid(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_std(data=ds.df, column_name=["time", "A"], window_size=1)
    assert result is None

# === Find Steady State Threshold ===
def test_find_steady_state_stationary(stationary_data):
    ds = DataStream(stationary_data)
    assert ds.find_steady_state_threshold(data=ds.df, column_name="A", window_size=2, threshold=0.1) == 2

def test_find_steady_state_long_data(long_data):
    ds = DataStream(long_data)
    assert ds.find_steady_state_threshold(data=ds.df, column_name="A", window_size=2, threshold=0.1) == 2

def test_find_steady_state_trim_data(trim_data):
    ds = DataStream(trim_data)
    assert ds.find_steady_state_threshold(data=ds.df, column_name="A", window_size=3, threshold=0.5) == 4

def test_find_steady_state_no_valid_data(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_threshold(data=ds.df, column_name="A", window_size=2, threshold=0.5)
    assert result is None

def test_find_steady_state_with_start_time(long_data):
    ds = DataStream(long_data)
    assert ds.find_steady_state_threshold(data=ds.df, column_name="A", window_size=2, threshold=0.1, start_time=1) == 3

# === Find Steady State Rolling Variance ===
def test_find_steady_state_rolling_variance_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.find_steady_state_rolling_variance(data=ds.df, column_name="A", window_size=3)
    assert result is None

def test_find_steady_state_none_rolling_variance(long_data):
    ds = DataStream(long_data)
    result = ds.find_steady_state_rolling_variance(data=long_data, column_name="A", window_size=3, threshold=0.1)
    assert result is None

def test_find_steady_state_rolling_variance_not_valid(no_valid_data):
    ds = DataStream(no_valid_data)
    result = ds.find_steady_state_rolling_variance(data=ds.df, column_name="A", window_size=1)
    assert result is None

# === effective_sample_size ===
def test_effective_sample_size_empty(empty_data):
    ds = DataStream(empty_data)
    assert ds.effective_sample_size() == {}

def test_effective_sample_size_nan(nan_data):
    ds = DataStream(nan_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert result["A"]["effective_sample_size"] is None
    assert result["A"]["message"] == "No data available for computation."

def test_effective_sample_size_simple(simple_data):
    ds = DataStream(simple_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert "A" in result and result["A"] is not None

def test_effective_sample_size_long_data(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size(column_names=["A", "B"])
    assert "A" in result and result["A"] is not None
    assert "B" in result and result["B"] is not None

def test_effective_sample_size_stationary(stationary_data):
    ds = DataStream(stationary_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert "A" in result and result["A"] is not None

def test_effective_sample_size_trim_data(trim_data):
    ds = DataStream(trim_data)
    result = ds.effective_sample_size(column_names=["A"])
    assert "A" in result and result["A"] is not None

def test_effective_sample_size_missing_col(long_data):
    ds = DataStream(long_data)
    result = ds.effective_sample_size(column_names=["C"])
    assert result["C"]["message"] == "Column 'C' not found in the DataStream."

