# Import statements
import numpy as np
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
    return pd.DataFrame({
        "A": [1, 2, 3]
    })

@pytest.fixture
def long_data():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
    })

# Test DataStream initialization
# =============================================================================

# Test initialization with simple data set
def test_init_simple(simple_data):
    ds = DataStream(simple_data)
    assert(len(ds) == 3)
    assert("A" in ds.variables())

# Test initialization with empty data set
def test_init_empty(empty_data):
    ds = DataStream(empty_data) # Does not throw an error
    assert(len(ds) == 0)
    assert(len(ds.variables()) == 0)

# Test mean
# =============================================================================
    
# Test mean with simple data set
def test_mean_simple(simple_data):
    ds = DataStream(simple_data)
    mean = ds.mean(window_size=1)
    expected = {'A': {'mean': 2.0}}
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
    expected = expected = {
        "A": {"mean": 3.0},
        "B": {"mean": 3.0}
    }
    assert mean == expected

# Test mean with overlapping windows
def test_mean_long_overlapping_window(long_data):
    ds = DataStream(long_data)
    mean = ds.mean()
    expected = {
        "A": {"mean": 3.0},
        "B": {"mean": 3.0}
    }
    print(mean)
    assert mean == expected

# Test mean with non-overlapping windows
def test_mean_long_non_overlapping_window(long_data):
    ds = DataStream(long_data)
    mean = ds.mean(method="non-overlapping", window_size=2)
    expected = {
        "A": {"mean": 3.0},
        "B": {"mean": 3.0}
    }
    assert mean == expected

# Test mean uncertainty
# =============================================================================

# Test mean uncertainty with simple data set
def test_mean_uncertainty_simple(simple_data):
    ds = DataStream(simple_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    expected = {
        'A': {'mean uncertainty': 0.5}
    }
    assert mean_uncertainty == expected

# Test mean uncertainty with long data set
def test_mean_uncertainty_long(long_data):
    ds = DataStream(long_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    expected = {
        'A': {'mean uncertainty': 0.6454972243679028}, 
        'B': {'mean uncertainty': 0.6454972243679028}
    }
    assert mean_uncertainty == expected

# Test Confidence Interval
# =============================================================================

def test_confidence_interval_simple(simple_data):
    ds = DataStream(simple_data)
    ci_df = ds.confidence_interval(window_size=1)
    expected = {
        'A': {'confidence interval': (0.8683934723883333, 3.131606527611667)}
    }
    assert ci_df == expected

def test_confidence_interval_long(long_data):
    ds = DataStream(long_data)
    ci_df = ds.confidence_interval(window_size=2)
    expected = {
        'A': {'confidence interval': (1.7348254402389105, 4.265174559761089)},
        'B': {'confidence interval': (1.7348254402389105, 4.265174559761089)}
    }
    assert ci_df == expected               