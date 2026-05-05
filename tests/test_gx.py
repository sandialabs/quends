import tempfile

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from quends import DataStream, from_gx


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
            heatflux_var[:] = np.linspace(15.0, 24.0, 10)
            wg_var[:] = np.linspace(1.0, 10.0, 10)
            wphi_var[:] = np.linspace(0.0, 270.0, 10)

        yield test_file


@pytest.fixture
def create_csv_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.csv"
        pd.DataFrame(
            {
                "time": range(10),
                "HeatFlux_st": [
                    20.5,
                    21.0,
                    19.5,
                    22.0,
                    23.5,
                    24.0,
                    25.0,
                    26.5,
                    27.0,
                    28.0,
                ],
                "Wg_st": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                "Wphi_st": [180, 190, 200, 210, 220, 230, 240, 250, 260, 270],
            }
        ).to_csv(test_file, index=False)
        yield test_file


def test_from_gx_netcdf_loads_single_requested_variable(create_netcdf_file):
    data_stream = from_gx(create_netcdf_file, variable="HeatFlux_st")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_series_equal(
        data_stream.data,
        pd.Series(
            np.linspace(15.0, 24.0, 10).astype(np.float32),
            name="HeatFlux_st",
        ),
    )


def test_from_gx_netcdf_loads_time_when_requested(create_netcdf_file):
    data_stream = from_gx(create_netcdf_file, variable="time")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_series_equal(
        data_stream.data,
        pd.Series(np.arange(10, dtype=np.float64), name="time"),
    )


def test_from_gx_without_variable_prints_guidance(create_netcdf_file, capsys):
    data_stream = from_gx(create_netcdf_file, variable=None)

    captured = capsys.readouterr()
    assert data_stream is None
    assert "No variable specified." in captured.out
    assert (
        "Please specify a variable to load only that variable "
        "(e.g. variable='temperature')."
    ) in captured.out


def test_from_gx_invalid_file():
    with pytest.raises(ValueError, match="Error: file .* does not exist."):
        from_gx("non_existent_file.nc", variable="HeatFlux_st")


def test_from_gx_unsupported_file_format():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"unsupported")
        temp_file.close()

        with pytest.raises(
            ValueError,
            match="Unsupported file format. Please provide a .nc or .csv file.",
        ):
            from_gx(temp_file.name, variable="HeatFlux_st")


def test_from_gx_csv_loads_single_requested_variable(create_csv_file):
    data_stream = from_gx(create_csv_file, variable="HeatFlux_st")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data,
        pd.DataFrame(
            {
                "HeatFlux_st": [
                    20.5,
                    21.0,
                    19.5,
                    22.0,
                    23.5,
                    24.0,
                    25.0,
                    26.5,
                    27.0,
                    28.0,
                ]
            }
        ),
    )


def test_from_gx_csv_loads_time_when_requested(create_csv_file):
    data_stream = from_gx(create_csv_file, variable="time")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data,
        pd.DataFrame({"time": list(range(10))}),
    )


def test_from_gx_no_variable(create_netcdf_file):
    data_stream = from_gx(create_netcdf_file, variable=None)

    assert data_stream is None
