# Tests for the single-variable from_dict loader.
import numpy as np
import pandas as pd
import pytest

import quends as qds


@pytest.fixture
def simple_dict():
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "HeatFlux_st": [10.0, 11.0, 12.0, 11.5, 12.5],
        }
    )


def test_from_dict_invalid_input():
    with pytest.raises(ValueError, match="Input must be a dictionary."):
        qds.from_dict("not_a_dict", "HeatFlux_st")


def test_from_dict_none_input():
    with pytest.raises(ValueError, match="Input must be a dictionary."):
        qds.from_dict(None, "HeatFlux_st")


def test_from_dict_missing_variable_raises():
    with pytest.raises(ValueError, match="does not exist"):
        qds.from_dict({}, "HeatFlux_st")


def test_from_dict_keeps_time_and_variable(simple_dict):
    data_dict = simple_dict.to_dict(orient="list")
    ds = qds.from_dict(data_dict, "HeatFlux_st")
    assert isinstance(ds, qds.DataStream)
    df = ds.data
    assert list(df.columns) == ["time", "HeatFlux_st"]
    np.testing.assert_array_equal(df["time"].values, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values, [10.0, 11.0, 12.0, 11.5, 12.5]
    )
