# Tests for the single-variable from_gx dispatcher (.nc -> netcdf, .csv -> csv).
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
            heatflux_var[:] = np.arange(15.0, 25.0)
            wg_var[:] = np.arange(0.0, 5.0, 0.5)
            wphi_var[:] = np.linspace(0.0, 360.0, 10)
            time_values = time_var[:]
            heatflux_values = heatflux_var[:]
        yield test_file, time_values, heatflux_values


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


def test_from_gx_netcdf_single_variable(create_netcdf_file):
    test_file, time_values, heatflux_values = create_netcdf_file
    ds = from_gx(test_file, "HeatFlux_st")
    assert isinstance(ds, DataStream)
    df = ds.data
    assert list(df.columns) == ["time", "HeatFlux_st"]
    assert "Wg_st" not in df.columns and "Wphi_st" not in df.columns
    assert len(df) == 10
    np.testing.assert_array_equal(df["time"].values, time_values)
    np.testing.assert_array_equal(df["HeatFlux_st"].values, heatflux_values)


def test_from_gx_csv_single_variable(create_csv_file):
    ds = from_gx(create_csv_file, "HeatFlux_st")
    assert isinstance(ds, DataStream)
    df = ds.data
    assert list(df.columns) == ["time", "HeatFlux_st"]
    assert "Wg_st" not in df.columns and "Wphi_st" not in df.columns
    assert len(df) == 10


def test_from_gx_no_variable_raises():
    with pytest.raises(ValueError, match="No variable specified"):
        from_gx("some_file.nc", "")


def test_from_gx_invalid_file():
    with pytest.raises(ValueError, match="does not exist"):
        from_gx("non_existent_file.nc", "HeatFlux_st")


def test_from_gx_unsupported_file_format():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a test file with an unsupported format.")
        temp_file.close()
        with pytest.raises(
            ValueError,
            match="Unsupported file format. Please provide a .nc or .csv file.",
        ):
            from_gx(temp_file.name, "HeatFlux_st")
