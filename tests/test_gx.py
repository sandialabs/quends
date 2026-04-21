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


def test_from_gx_without_variables_returns_default_stream_mapping(create_netcdf_file):
    streams = from_gx(create_netcdf_file)

    assert set(streams) == {"time", "HeatFlux_st", "Wg_st", "Wphi_st"}
    assert all(isinstance(stream, DataStream) for stream in streams.values())
    assert streams["time"].data.name == "time"
    assert streams["HeatFlux_st"].data.name == "HeatFlux_st"


def test_from_gx_with_variables_returns_requested_stream_mapping(create_netcdf_file):
    streams = from_gx(create_netcdf_file, variables=["time", "HeatFlux_st"])

    assert set(streams) == {"time", "HeatFlux_st"}
    assert all(isinstance(stream, DataStream) for stream in streams.values())


def test_from_gx_invalid_file():
    with pytest.raises(ValueError, match="Error: file .* does not exist."):
        from_gx("non_existent_file.nc")


def test_from_gx_unsupported_file_format():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"unsupported")
        temp_file.close()

        with pytest.raises(
            ValueError,
            match="Unsupported file format. Please provide a .nc or .csv file.",
        ):
            from_gx(temp_file.name)


def test_from_gx_csv_without_variables_returns_default_stream_mapping(create_csv_file):
    streams = from_gx(create_csv_file)

    assert set(streams) == {"time", "HeatFlux_st", "Wg_st", "Wphi_st"}
    assert all(isinstance(stream, DataStream) for stream in streams.values())
    pd.testing.assert_frame_equal(
        streams["time"].data, pd.DataFrame({"time": list(range(10))})
    )


def test_from_gx_csv_with_specific_variables_returns_requested_stream_mapping(
    create_csv_file,
):
    streams = from_gx(create_csv_file, variables=["time", "HeatFlux_st"])

    assert set(streams) == {"time", "HeatFlux_st"}
    assert all(isinstance(stream, DataStream) for stream in streams.values())
