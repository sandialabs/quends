# Tests for the single-variable from_csv loader.
import tempfile

import numpy as np
import pandas as pd
import pytest

from quends import DataStream, from_csv


@pytest.fixture
def create_csv_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.csv"
        data = {
            "time": range(10),
            "HeatFlux_st": [20.5, 21.0, 19.5, 22.0, 23.5, 24.0, 25.0, 26.5, 27.0, 28.0],
            "Wg_st": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            "Wphi_st": [180, 190, 200, 210, 220, 230, 240, 250, 260, 270],
        }
        pd.DataFrame(data).to_csv(test_file, index=False)
        yield test_file


def test_from_csv_keeps_time_and_variable(create_csv_file):
    """Single-variable load returns exactly [time, variable]."""
    ds = from_csv(create_csv_file, "HeatFlux_st")
    assert isinstance(ds, DataStream)
    assert hasattr(ds, "data")
    df = ds.data

    assert list(df.columns) == ["time", "HeatFlux_st"]
    # Other variables are not loaded.
    assert "Wg_st" not in df.columns
    assert "Wphi_st" not in df.columns
    assert len(df) == 10

    np.testing.assert_array_equal(df["time"].values, np.arange(10))
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values,
        [20.5, 21.0, 19.5, 22.0, 23.5, 24.0, 25.0, 26.5, 27.0, 28.0],
    )


def test_from_csv_missing_variable_raises(create_csv_file):
    """Requesting a column that does not exist raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        from_csv(create_csv_file, "NotAColumn")


def test_from_csv_non_existent_file():
    with pytest.raises(ValueError, match="does not exist"):
        from_csv("non_existent_file.csv", "HeatFlux_st")


def test_from_csv_records_provenance(create_csv_file):
    """Loader records a 'load' history entry with source/variable/time provenance."""
    ds = from_csv(create_csv_file, "HeatFlux_st")
    entries = ds.history.entries()
    load = [e for e in entries if e.operation_name == "load"]
    assert load, "expected a 'load' history entry"
    params = load[0].parameters
    assert params["loader"] == "from_csv"
    assert params["variable"] == "HeatFlux_st"
    assert params["time_column"] == "time"
    assert params["time_resolution"] == "name_alias"
    assert create_csv_file in params["source"]
