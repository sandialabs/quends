# Import statements
import tempfile

import numpy as np
import pandas as pd
import pytest

# Special imports
from quends import DataStream, from_csv


# Fixture to create a CSV file in a temporary directory
# =============================================================================
@pytest.fixture
def create_csv_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.csv"

        # Create sample data
        data = {
            "time": range(10),
            "HeatFlux_st": [20.5, 21.0, 19.5, 22.0, 23.5, 24.0, 25.0, 26.5, 27.0, 28.0],
            "Wg_st": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            "Wphi_st": [180, 190, 200, 210, 220, 230, 240, 250, 260, 270],
        }
        df = pd.DataFrame(data)
        df.to_csv(test_file, index=False)  # Save DataFrame to CSV without the index

        yield test_file  # Yield the path to the created CSV file


# Test 'from_csv' with no variables assigned
# =============================================================================
def test_from_csv_without_variables(create_csv_file):
    """Test reading a CSV file without specifying variables."""
    test_file = create_csv_file  # Get the filename from the fixture

    # Call from_csv without passing any variables
    data_stream = from_csv(test_file)  # No variables passed

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(data_stream, "df"), "DataStream does not have a 'df' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.df

    # Check if the DataFrame contains all expected columns
    expected_columns = ["time", "HeatFlux_st", "Wg_st", "Wphi_st"]
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."


# Test 'from_csv' with variables assigned
# =============================================================================
def test_from_csv_with_variables(create_csv_file):
    """Test reading a CSV file with specific columns."""
    test_file = create_csv_file  # Get the filename from the fixture

    # Call from_csv to read the CSV file, specifying columns to load
    data_stream = from_csv(test_file, variables=["time", "HeatFlux_st"])

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(data_stream, "df"), "DataStream does not have a 'df' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.df

    # Check if the DataFrame contains the expected columns
    assert "time" in df.columns, "DataFrame should contain 'time' column."
    assert "HeatFlux_st" in df.columns, "DataFrame should contain 'HeatFlux_st' column."

    # Ensure that unwanted variables are not present
    assert "Wg_st" not in df.columns, "DataFrame should not contain 'Wg_st' column."
    assert "Wphi_st" not in df.columns, "DataFrame should not contain 'Wphi_st' column."

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."

    # Verify that the data values match what was written
    expected_data = {
        "time": range(10),
        "HeatFlux_st": [20.5, 21.0, 19.5, 22.0, 23.5, 24.0, 25.0, 26.5, 27.0, 28.0],
    }
    np.testing.assert_array_equal(
        df["time"].values, expected_data["time"], "Time values do not match."
    )
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values,
        expected_data["HeatFlux_st"],
        "HeatFlux_st values do not match.",
    )
    # Test loading from a CSV file without specifying variables


def test_from_csv_non_existent_file():
    """Test that from_csv raises a ValueError for a non-existent file."""
    # Define a path for a non-existent CSV file
    non_existent_file = "non_existent_file.csv"

    # Use pytest.raises to check for ValueError
    with pytest.raises(
        ValueError, match=f"Error: file {non_existent_file} does not exist."
    ):
        from_csv(non_existent_file)
