import json
import os

import numpy as np
import pytest

from quends import (  # Replace with the actual import for your DataStream class
    DataStream,
    from_json,
)


# Fixture to create a temporary JSON file
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
    os.remove(test_file)  # Clean up the file after the test


# Test loading a valid JSON file
def test_from_json_valid(create_json_file):
    test_file = create_json_file

    # Call from_json to read the JSON file
    data_stream = from_json(test_file)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "df"
    ), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.df

    # Check if the DataFrame contains all expected columns
    expected_columns = ["time", "HeatFlux_st", "Wg_st", "Wphi_st"]
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."

    # Verify that the data values match what was written
    np.testing.assert_array_equal(
        df["time"].values, [0, 1, 2, 3, 4], "Time values do not match."
    )
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values,
        [15.0, 16.5, 18.0, 19.5, 21.0],
        "HeatFlux_st values do not match.",
    )
    np.testing.assert_array_equal(
        df["Wg_st"].values, [0.5, 1.0, 1.5, 2.0, 2.5], "Wg_st values do not match."
    )
    np.testing.assert_array_equal(
        df["Wphi_st"].values,
        [0.0, 45.0, 90.0, 135.0, 180.0],
        "Wphi_st values do not match.",
    )

    # Validate the contents of the DataFrame
    assert len(df) == 5, "DataFrame should have 5 entries."


# Test loading specific variables
def test_from_json_with_specific_variables(create_json_file):
    test_file = create_json_file

    # Specify the variables to read
    variables_to_read = ["HeatFlux_st", "Wg_st"]

    # Call from_json with specific variables
    data_stream = from_json(test_file, variables=variables_to_read)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "df"
    ), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.df

    # Check if the DataFrame contains only the expected columns
    expected_columns = ["HeatFlux_st", "Wg_st"]
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."

    # Check that the DataFrame does not contain any unexpected columns
    unexpected_columns = ["time", "Wphi_st"]
    for column in unexpected_columns:
        assert (
            column not in df.columns
        ), f"DataFrame should not contain '{column}' column."

    # Verify that the data values match what was written
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values,
        [15.0, 16.5, 18.0, 19.5, 21.0],
        "HeatFlux_st values do not match.",
    )
    np.testing.assert_array_equal(
        df["Wg_st"].values, [0.5, 1.0, 1.5, 2.0, 2.5], "Wg_st values do not match."
    )

    # Validate the contents of the DataFrame
    assert len(df) == 5, "DataFrame should have 5 entries."


# Test loading a non-existent JSON file
def test_from_json_non_existent_file():
    # Define a path for a non-existent JSON file
    non_existent_file = "non_existent_file.json"

    # Use pytest.raises to check for ValueError
    with pytest.raises(
        ValueError, match=f"Error: file {non_existent_file} does not exist."
    ):
        from_json(non_existent_file)


# Test loading an invalid JSON file
def test_from_json_invalid_file():
    # Create an invalid JSON file
    invalid_json_file = "invalid_data.json"
    with open(invalid_json_file, "w") as f:
        f.write("This is not a valid JSON.")

    # Use pytest.raises to check for ValueError
    with pytest.raises(ValueError):
        from_json(invalid_json_file)

    # Clean up the invalid file
    os.remove(invalid_json_file)
