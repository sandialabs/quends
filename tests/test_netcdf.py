import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from quends import DataStream, from_netcdf


@pytest.fixture
def create_netcdf_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            dataset.createDimension("time", None)
            grids_group = dataset.createGroup("Grids")
            diagnostics_group = dataset.createGroup("Diagnostics")

            time_var = grids_group.createVariable("time", np.float64, ("time",))
            heatflux_var = diagnostics_group.createVariable(
                "HeatFlux_st", np.float32, ("time",)
            )
            wg_var = diagnostics_group.createVariable("Wg_st", np.float32, ("time",))
            wphi_var = diagnostics_group.createVariable(
                "Wphi_st", np.float32, ("time",)
            )

            time_var[:] = np.arange(10)
            heatflux_var[:] = np.array(
                [15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]
            )
            wg_var[:] = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            wphi_var[:] = np.array(
                [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 360.0]
            )

        yield test_file


@pytest.fixture
def create_netcdf_file_with_truncated_2d_diagnostic():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            dataset.createDimension("time", None)
            dataset.createDimension("dim2", 3)
            grids_group = dataset.createGroup("Grids")
            diagnostics_group = dataset.createGroup("Diagnostics")

            time_var = grids_group.createVariable("time", np.float64, ("time",))
            heatflux_var = diagnostics_group.createVariable(
                "HeatFlux_st", np.float32, ("time",)
            )
            truncated_flux_t = diagnostics_group.createVariable(
                "TruncatedFlux_t", np.float32, ("time", "dim2")
            )

            time_var[:] = np.arange(10)
            heatflux_var[:] = np.array(
                [15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]
            )
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

        yield test_file


@pytest.fixture
def create_netcdf_file_for_diagnostic_shape_cases():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/branch_coverage.nc"

        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            dataset.createDimension("time", 10)
            dataset.createDimension("short_time", 4)
            dataset.createDimension("rows", 2)
            dataset.createDimension("cols_exact", 5)
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
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
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


def test_from_netcdf_loads_requested_variable_as_series(create_netcdf_file):
    data_stream = from_netcdf(create_netcdf_file, variable="HeatFlux_st")

    assert isinstance(data_stream, DataStream)
    assert isinstance(data_stream.data, pd.Series)
    np.testing.assert_array_equal(
        data_stream.data.values,
        np.array([15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]),
    )


def test_from_netcdf_loads_time_variable_as_series(create_netcdf_file):
    data_stream = from_netcdf(create_netcdf_file, variable="time")

    assert isinstance(data_stream, DataStream)
    assert isinstance(data_stream.data, pd.Series)
    np.testing.assert_array_equal(data_stream.data.values, np.arange(10))


def test_from_netcdf_non_existent_file():
    non_existent_file = "non_existent_file.nc"

    with pytest.raises(
        ValueError, match=f"Error: file {non_existent_file} does not exist."
    ):
        from_netcdf(non_existent_file, variable="HeatFlux_st")


def test_from_netcdf_missing_variable_raises(create_netcdf_file):
    with pytest.raises(
        ValueError, match="Error: variable 'missing' does not exist in file"
    ):
        from_netcdf(create_netcdf_file, variable="missing")


def test_from_netcdf_truncates_flattened_2d_diagnostic(
    create_netcdf_file_with_truncated_2d_diagnostic,
):
    data_stream = from_netcdf(
        create_netcdf_file_with_truncated_2d_diagnostic,
        variable="TruncatedFlux_t",
    )

    assert isinstance(data_stream, DataStream)
    np.testing.assert_array_equal(
        data_stream.data.values,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )


def test_from_netcdf_expands_scalar_and_pads_short_1d(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    scalar_stream = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, variable="scalar_st"
    )
    padded_stream = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, variable="PaddedSignal_st"
    )

    np.testing.assert_array_equal(scalar_stream.data.values, np.full(10, 3.5))
    np.testing.assert_allclose(
        padded_stream.data.values,
        np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )


def test_from_netcdf_flattens_exact_length_and_padded_2d_diagnostics(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    exact_stream = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, variable="ExactLengthSignal_t"
    )
    padded_stream = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, variable="PaddedMatrix_st"
    )

    np.testing.assert_array_equal(
        exact_stream.data.values,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )
    np.testing.assert_allclose(
        padded_stream.data.values,
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )


def test_from_netcdf_ignores_diagnostics_without_supported_suffix(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    with pytest.raises(
        ValueError, match="Error: variable 'IgnoredMatrix' does not exist in file"
    ):
        from_netcdf(
            create_netcdf_file_for_diagnostic_shape_cases, variable="IgnoredMatrix"
        )

    with pytest.raises(
        ValueError, match="Error: variable 'UnsupportedCube_t' does not exist in file"
    ):
        from_netcdf(
            create_netcdf_file_for_diagnostic_shape_cases, variable="UnsupportedCube_t"
        )
