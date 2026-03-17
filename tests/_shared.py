import numpy as np
import pandas as pd
import pytest


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
def partial_nan_data():
    return pd.DataFrame({"A": [1, 2, 3], "B": [None, None, None]})


@pytest.fixture
def no_valid_data():
    return pd.DataFrame({"time": [0, 1], "A": [1, 2]})


@pytest.fixture
def stationary_noise_df():
    np.random.seed(0)
    return pd.DataFrame(
        {
            "time": np.arange(300),
            "A": np.random.normal(0, 1, 300),
        }
    )


@pytest.fixture
def slope_to_stationary_df():
    np.random.seed(42)
    trend = 2 * np.arange(100)
    stationary = np.random.normal(0, 5, 400)
    signal = np.concatenate([trend, stationary])
    return pd.DataFrame({"time": np.arange(len(signal)), "A": signal})


@pytest.fixture
def persistent_trend_df():
    np.random.seed(123)
    x = np.arange(500)
    signal = 5 * x + np.random.normal(0, 10, 500)
    return pd.DataFrame({"time": x, "A": signal})


@pytest.fixture
def intermittent_stationary_df():
    np.random.seed(789)
    n = 300
    base_signal = np.random.normal(0, 1, n)
    spike_count = 20
    spike_indices = np.random.choice(range(50, n), size=spike_count, replace=False)
    base_signal[spike_indices] += np.random.uniform(50, 100, spike_count)
    return pd.DataFrame({"time": np.arange(n), "A": base_signal})


@pytest.fixture
def high_frequency_noise_df():
    np.random.seed(555)
    t = np.arange(400)
    signal = 10 * np.sin(t * 0.5) + np.random.normal(50, 2, 400)
    return pd.DataFrame({"time": t, "A": signal})


@pytest.fixture
def oscillating_to_stable_df():
    np.random.seed(456)
    t = np.arange(400)
    oscillation = 20 * np.exp(-t / 100) * np.sin(t / 10)
    noise = np.random.normal(0, 1, 400)
    signal = oscillation + noise + 50
    return pd.DataFrame({"time": t, "A": signal})


@pytest.fixture
def multiple_transitions_df():
    np.random.seed(999)
    trend1 = np.linspace(0, 50, 100)
    plateau1 = np.random.normal(50, 2, 100)
    trend2 = np.linspace(50, 80, 100)
    plateau2 = np.random.normal(80, 2, 200)
    signal = np.concatenate([trend1, plateau1, trend2, plateau2])
    return pd.DataFrame({"time": np.arange(len(signal)), "A": signal})
