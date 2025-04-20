import pandas as pd
import pytest

import quends as qds


@pytest.fixture
def simple_dict():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
        }
    )


# Test dictionary for simple_dict()
# =============================================================================
def test_from_dict_invalid_input():
    with pytest.raises(ValueError, match="Input must be a dictionary."):
        qds.from_dict("not_a_dict")  # Passing a string instead of a dictionary


def test_from_dict_empty_dict():
    data_dict = {}
    data_stream = qds.from_dict(data_dict)
    assert isinstance(data_stream, qds.DataStream), "Expected a DataStream object"
    df = data_stream.df
    assert df.empty, "DataFrame should be empty for an empty dictionary."


def test_from_dict_none_input():
    with pytest.raises(ValueError, match="Input must be a dictionary."):
        qds.from_dict(None)  # Passing None instead of a dictionary


def test_from_dict_with_none_variables(simple_dict):
    data_dict = simple_dict.to_dict(orient="list")
    data_stream = qds.from_dict(data_dict, variables=None)  # Should include all columns
    df = data_stream.df
    expected_columns = ["A", "B"]
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."
