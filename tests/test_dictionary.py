import pandas as pd
import pytest

import quends as qds


@pytest.fixture
def simple_dict():
    return {"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}


def test_from_dict_invalid_input():
    with pytest.raises(ValueError, match="Input must be a dictionary."):
        qds.from_dict("not_a_dict", variable="A")


def test_from_dict_none_input():
    with pytest.raises(ValueError, match="Input must be a dictionary."):
        qds.from_dict(None, variable="A")


def test_from_dict_loads_single_requested_variable(simple_dict):
    data_stream = qds.from_dict(simple_dict, variable="A")

    assert isinstance(data_stream, qds.DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data, pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    )


def test_from_dict_missing_variable_raises(simple_dict):
    with pytest.raises(
        ValueError, match="Error: variable 'missing' does not exist in the dictionary."
    ):
        qds.from_dict(simple_dict, variable="missing")
