# Import statements
import tempfile

import netCDF4 as nc
import numpy as np
import pytest

# Special imports
from quends import DataStream, from_netcdf


# Fixture to create a NetCDF file in a temporary directory
# =============================================================================
@pytest.fixture
def create_netcdf_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            # Create dimensions
            dataset.createDimension("time", None)

            # Create the "Grids" group
            grids_group = dataset.createGroup("Grids")

            # Create the "Diagnostics" group
            diagnostics_group = dataset.createGroup("Diagnostics")

            # Create variables in the "Grids" group
            time_var = grids_group.createVariable("time", np.float64, ("time",))
            heatflux_var = diagnostics_group.createVariable(
                "HeatFlux_st", np.float32, ("time",)
            )
            wg_var = diagnostics_group.createVariable("Wg_st", np.float32, ("time",))
            wphi_var = diagnostics_group.createVariable(
                "Wphi_st", np.float32, ("time",)
            )

            time_var[:] = np.arange(10)  # Time values from 0 to 9
            heatflux_var[:] = np.array(
                [15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]
            )  # Fixed heat flux values
            wg_var[:] = np.array(
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            )  # Fixed wg values
            wphi_var[:] = np.array(
                [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 360.0]
            )  # Fixed wphi values

            # Read back the values from the NetCDF variables
            time_values = time_var[:]
            heatflux_values = heatflux_var[:]
            wg_values = wg_var[:]
            wphi_values = wphi_var[:]
        yield test_file, time_values, heatflux_values, wg_values, wphi_values  # Yield the path and original values


# Test 'from_netcdf' with no variables assigned
# =============================================================================
def test_from_netcdf_without_variables(create_netcdf_file):
    test_file, time_values, heatflux_values, wg_values, wphi_values = create_netcdf_file

    # Call from_netcdf without passing any variables
    data_stream = from_netcdf(test_file)  # No variables passed

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "data"
    ), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.data

    # Check if the DataFrame contains all expected columns
    expected_columns = ["time", "HeatFlux_st", "Wg_st", "Wphi_st"]
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."

    # Verify that the data values match what was written
    np.testing.assert_array_equal(
        df["time"].values, time_values, "Time values do not match."
    )
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values, heatflux_values, "HeatFlux_st values do not match."
    )
    np.testing.assert_array_equal(
        df["Wg_st"].values, wg_values, "Wg_st values do not match."
    )
    np.testing.assert_array_equal(
        df["Wphi_st"].values, wphi_values, "Wphi_st values do not match."
    )

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."


# Test 'from_netcdf' with nonexisting file
# =============================================================================
def test_from_netcdf_non_existent_file():
    # Define a path for a non-existent NetCDF file
    non_existent_file = "non_existent_file.nc"

    # Use pytest.raises to check for ValueError
    with pytest.raises(
        ValueError, match=f"Error: file {non_existent_file} does not exist."
    ):
        from_netcdf(non_existent_file)


# Test 'from_netcdf' with specific variables assigned
def test_from_netcdf_with_specific_variables(create_netcdf_file):
    test_file, time_values, heatflux_values, wg_values, wphi_values = (
        create_netcdf_file  # This will now be just the file path
    )

    # Specify the variables to read
    variables_to_read = ["HeatFlux_st", "Wg_st"]

    # Call from_netcdf with specific variables
    data_stream = from_netcdf(test_file, variables=variables_to_read)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "data"
    ), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.data

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
        np.array([15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]),
        "HeatFlux_st values do not match.",
    )
    np.testing.assert_array_equal(
        df["Wg_st"].values,
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        "Wg_st values do not match.",
    )

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."


# Fixture to create a NetCDF file in a temporary directory
@pytest.fixture
def create_another_netcdf_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            # Create dimensions
            dataset.createDimension("time", None)
            dataset.createDimension(
                "dim2", 2
            )  # Create a second dimension with a fixed size

            # Create the "Grids" group
            grids_group = dataset.createGroup("Grids")

            # Create the "Diagnostics" group
            diagnostics_group = dataset.createGroup("Diagnostics")

            # Create variables in the "Grids" group
            time_var = grids_group.createVariable("time", np.float64, ("time",))
            heatflux_var = diagnostics_group.createVariable(
                "HeatFlux_st", np.float32, ("time",)
            )
            wg_var = diagnostics_group.createVariable("Wg_st", np.float32, ("time",))
            wphi_var = diagnostics_group.createVariable(
                "Wphi_st", np.float32, ("time",)
            )

            # Assign fixed data to variables
            time_var[:] = np.arange(10)  # Time values from 0 to 9
            heatflux_var[:] = np.array(
                [15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]
            )  # Fixed heat flux values
            wg_var[:] = np.array(
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            )  # Fixed wg values
            wphi_var[:] = np.array(
                [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 360.0]
            )  # Fixed wphi values

        yield test_file  # Yield only the path to the created NetCDF file


# Test 'from_netcdf' with specific variables assigned
def test_from_netcdf_with_second_netcdf(create_another_netcdf_file):
    test_file = create_another_netcdf_file  # This will now be just the file path

    # Specify the variables to read
    variables_to_read = ["HeatFlux_st", "Wg_st"]

    # Call from_netcdf with specific variables
    data_stream = from_netcdf(test_file, variables=variables_to_read)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "data"
    ), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.data

    # Verify that the data values match what was written
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values,
        np.array([15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]),
        "HeatFlux_st values do not match.",
    )
    np.testing.assert_array_equal(
        df["Wg_st"].values,
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        "Wg_st values do not match.",
    )

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."


# Fixture to create a NetCDF file in a temporary directory with a truncation case
@pytest.fixture
def create_netcdf_file_with_truncated_2d_diagnostic():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            # Create dimensions
            dataset.createDimension("time", None)
            dataset.createDimension(
                "dim2", 3
            )  # Create a second dimension with a fixed size

            # Create the "Grids" group
            grids_group = dataset.createGroup("Grids")

            # Create the "Diagnostics" group
            diagnostics_group = dataset.createGroup("Diagnostics")

            # Create variables in the "Grids" group
            time_var = grids_group.createVariable("time", np.float64, ("time",))
            heatflux_var = diagnostics_group.createVariable(
                "HeatFlux_st", np.float32, ("time",)
            )
            wg_var = diagnostics_group.createVariable("Wg_st", np.float32, ("time",))
            wphi_var = diagnostics_group.createVariable(
                "Wphi_st", np.float32, ("time",)
            )

            # Create a 2D variable in the Diagnostics group
            truncated_flux_t = diagnostics_group.createVariable(
                "TruncatedFlux_t", np.float32, ("time", "dim2")
            )

            # Assign fixed data to variables
            time_var[:] = np.arange(10)  # Time values from 0 to 9
            heatflux_var[:] = np.array(
                [15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]
            )  # Fixed heat flux values
            wg_var[:] = np.array(
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            )  # Fixed wg values
            wphi_var[:] = np.array(
                [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 360.0]
            )  # Fixed wphi values

            # Assign fixed data to the 2D variable
            truncated_flux_t[:, :] = np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0],
                    [19.0, 20.0, 21.0],
                    [22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0],
                ]
            )

        yield test_file  # Yield only the path to the created NetCDF file


def test_from_netcdf_truncates_flattened_2d_diagnostic(
    create_netcdf_file_with_truncated_2d_diagnostic,
):
    test_file = create_netcdf_file_with_truncated_2d_diagnostic

    # Specify the variables to read
    variables_to_read = ["HeatFlux_st", "Wg_st", "TruncatedFlux_t"]

    # Call from_netcdf with specific variables
    data_stream = from_netcdf(test_file, variables=variables_to_read)

    # Check if the result is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object."

    # Check if DataStream has a df attribute
    assert hasattr(
        data_stream, "data"
    ), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.data
    # Verify that the data values match what was written
    np.testing.assert_array_equal(
        df["HeatFlux_st"].values,
        np.array([15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]),
        "HeatFlux_st values do not match.",
    )
    np.testing.assert_array_equal(
        df["Wg_st"].values,
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        "Wg_st values do not match.",
    )
    np.testing.assert_array_equal(
        df["TruncatedFlux_t"].values,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )
    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."


@pytest.fixture
def create_netcdf_file_for_diagnostic_shape_cases():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/branch_coverage.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            dataset.createDimension("time", 10)
            dataset.createDimension("short_time", 4)
            dataset.createDimension("rows", 2)
            dataset.createDimension("cols_exact", 5)
            dataset.createDimension("cols_long", 6)
            dataset.createDimension("cols_short", 3)
            dataset.createDimension("depth", 2)

            grids_group = dataset.createGroup("Grids")
            diagnostics_group = dataset.createGroup("Diagnostics")

            time_var = grids_group.createVariable("time", np.float64, ("time",))
            scalar_var = diagnostics_group.createVariable("scalar_st", np.float32)
            padded_signal_st = diagnostics_group.createVariable(
                "PaddedSignal_st", np.float32, ("short_time",)
            )
            exact_length_2d_signal_t = diagnostics_group.createVariable(
                "ExactLengthSignal_t", np.float32, ("rows", "cols_exact")
            )
            padded_2d_signal_st = diagnostics_group.createVariable(
                "PaddedMatrix_st", np.float32, ("rows", "cols_short")
            )
            unsupported_cube_t = diagnostics_group.createVariable(
                "UnsupportedCube_t", np.float32, ("depth", "rows", "cols_short")
            )
            ignored_diagnostic = diagnostics_group.createVariable(
                "IgnoredMatrix", np.float32, ("rows", "cols_short")
            )

            time_var[:] = np.arange(10)
            scalar_var.assignValue(3.5)
            padded_signal_st[:] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            exact_length_2d_signal_t[:, :] = np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                ],
                dtype=np.float32,
            )
            padded_2d_signal_st[:, :] = np.array(
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
            )
            unsupported_cube_t[:, :, :] = np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            )
            ignored_diagnostic[:, :] = np.array(
                [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]], dtype=np.float32
            )

        yield test_file


def test_from_netcdf_expands_scalar_and_pads_short_1d(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    data_stream = from_netcdf(create_netcdf_file_for_diagnostic_shape_cases)
    df = data_stream.data

    np.testing.assert_array_equal(df["time"].values, np.arange(10))
    np.testing.assert_array_equal(df["scalar_st"].values, np.full(10, 3.5))

    expected_short_series = np.array(
        [1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        df["PaddedSignal_st"].values,
        expected_short_series,
        equal_nan=True,
    )


def test_from_netcdf_flattens_exact_length_and_padded_2d_diagnostics(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    data_stream = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases,
        variables=["ExactLengthSignal_t", "PaddedMatrix_st"],
    )
    df = data_stream.data

    np.testing.assert_array_equal(
        df["ExactLengthSignal_t"].values,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )

    expected_short_matrix = np.array(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, np.nan, np.nan, np.nan, np.nan]
    )
    np.testing.assert_allclose(
        df["PaddedMatrix_st"].values,
        expected_short_matrix,
        equal_nan=True,
    )


def test_from_netcdf_ignores_diagnostics_without_supported_suffix(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    data_stream = from_netcdf(create_netcdf_file_for_diagnostic_shape_cases)
    df = data_stream.data

    assert "IgnoredMatrix" not in df.columns
    assert "UnsupportedCube_t" not in df.columns
