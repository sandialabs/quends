# Tests for the single-variable from_json loader.
import json
import os

import numpy as np
import pytest

from quends import DataStream, from_json


@pytest.fixture
def create_json_file():
    test_data = {
        "time": [0, 1, 2, 3, 4],
        "HeatFlux_st": [15.0, 16.5, 18.0, 19.5, 21.0],
        "Wg_st": [0.5, 1.0, 1.5, 2.0, 2.5],
        "Wphi_st": [0.0, 45.0, 90.0, 135.0, 180.0],
    }
    test_file = "test_data.json"
    with open(test_file, "w") as f:
        json.dump(test_data, f)
    yield test_file
    os.remove(test_file)


def test_from_json_keeps_time_and_variable(create_json_file):
    ds = from_json(create_json_file, "HeatFlux_st")
    assert isinstance(ds, DataStream)
    df = ds.data

    assert list(df.columns) == ["time", "HeatFlux_st"]
    assert "Wg_st" not in df.columns
    assert "Wphi_st" not in df.columns
    assert len(df) == 5

    np.testing.assert_array_equal(df["time"].values, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values, [15.0, 16.5, 18.0, 19.5, 21.0]
    )


def test_from_json_missing_variable_raises(create_json_file):
    with pytest.raises(ValueError, match="does not exist"):
        from_json(create_json_file, "NotAColumn")


def test_from_json_non_existent_file():
    with pytest.raises(ValueError, match="does not exist"):
        from_json("non_existent_file.json", "HeatFlux_st")


def test_from_json_invalid_file():
    invalid_json_file = "invalid_data.json"
    with open(invalid_json_file, "w") as f:
        f.write("This is not a valid JSON.")
    try:
        with pytest.raises(ValueError):
            from_json(invalid_json_file, "HeatFlux_st")
    finally:
        os.remove(invalid_json_file)
