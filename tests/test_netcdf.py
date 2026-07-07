# Tests for the single-variable from_netcdf loader.
# from_netcdf still extracts/reshapes all _t/_st diagnostics internally, then
# returns exactly [time, variable]; these tests request one variable per call.
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

from quends import DataStream, from_netcdf

SMALL_NETCDF_FILE = (
    Path(__file__).resolve().parents[1] / "examples" / "data" / "test" / "small.nc"
)


@pytest.fixture
def small_netcdf_file():
    assert SMALL_NETCDF_FILE.exists(), f"Missing test fixture: {SMALL_NETCDF_FILE}"
    return SMALL_NETCDF_FILE


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
            time_values = time_var[:]
            heatflux_values = heatflux_var[:]
            wg_values = wg_var[:]
        yield test_file, time_values, heatflux_values, wg_values


def test_from_netcdf_keeps_time_and_variable(create_netcdf_file):
    test_file, time_values, heatflux_values, _ = create_netcdf_file
    ds = from_netcdf(test_file, "HeatFlux_st")
    assert isinstance(ds, DataStream)
    df = ds.data
    assert list(df.columns) == ["time", "HeatFlux_st"]
    assert "Wg_st" not in df.columns and "Wphi_st" not in df.columns
    assert len(df) == 10
    np.testing.assert_array_equal(df["time"].values, time_values)
    np.testing.assert_array_equal(df["HeatFlux_st"].values, heatflux_values)


def test_from_netcdf_other_variable(create_netcdf_file):
    test_file, _, _, wg_values = create_netcdf_file
    df = from_netcdf(test_file, "Wg_st").data
    assert list(df.columns) == ["time", "Wg_st"]
    np.testing.assert_array_equal(df["Wg_st"].values, wg_values)


def test_from_netcdf_reads_persistent_small_fixture(small_netcdf_file):
    ds = from_netcdf(small_netcdf_file, "HeatFlux_st")

    assert isinstance(ds, DataStream)
    assert list(ds.data.columns) == ["time", "HeatFlux_st"]
    np.testing.assert_array_equal(ds.data["time"].values, np.arange(10))
    np.testing.assert_array_equal(
        ds.data["HeatFlux_st"].values,
        np.array([15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 24.0, 25.5, 27.0, 28.5]),
    )


def test_from_netcdf_persistent_small_fixture_other_variable(small_netcdf_file):
    ds = from_netcdf(small_netcdf_file, "Wphi_st")

    assert list(ds.data.columns) == ["time", "Wphi_st"]
    np.testing.assert_array_equal(
        ds.data["Wphi_st"].values,
        np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0, 360.0]),
    )


def test_small_netcdf_loaded_stream_can_be_normalized(small_netcdf_file):
    ds = from_netcdf(small_netcdf_file, "HeatFlux_st")

    normalized = DataStream.normalize_data(ds.data)

    assert list(normalized.columns) == ["time", "HeatFlux_st"]
    np.testing.assert_array_equal(normalized["time"].values, np.arange(10))
    np.testing.assert_allclose(
        normalized["HeatFlux_st"].values,
        np.linspace(0.0, 1.0, 10),
    )


def test_small_netcdf_loaded_stream_can_compute_statistics(small_netcdf_file):
    ds = from_netcdf(small_netcdf_file, "HeatFlux_st")

    stats = ds.compute_statistics("HeatFlux_st", window_size=2)["HeatFlux_st"]

    assert stats["mean"] == pytest.approx(21.75)
    assert stats["variance"] == pytest.approx(22.5)
    assert stats["standard_deviation"] == pytest.approx(np.sqrt(22.5))
    assert stats["window_size"] == 2
    assert stats["n_short_averages"] == 5
    assert stats["se_method"] == "ess_blocks"


def test_from_netcdf_missing_variable_raises(create_netcdf_file):
    test_file, *_ = create_netcdf_file
    with pytest.raises(ValueError, match="does not exist"):
        from_netcdf(test_file, "NotADiagnostic")


def test_from_netcdf_non_existent_file():
    with pytest.raises(ValueError, match="does not exist"):
        from_netcdf("non_existent_file.nc", "HeatFlux_st")


# --- 2D diagnostic flatten/truncate ----------------------------------------
@pytest.fixture
def create_netcdf_file_with_truncated_2d_diagnostic():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"
        with nc.Dataset(test_file, "w", format="NETCDF4") as dataset:
            dataset.createDimension("time", None)
            dataset.createDimension("dim2", 3)
            grids_group = dataset.createGroup("Grids")
            diagnostics_group = dataset.createGroup("Diagnostics")
            grids_group.createVariable("time", np.float64, ("time",))[:] = np.arange(10)
            truncated_flux_t = diagnostics_group.createVariable(
                "TruncatedFlux_t", np.float32, ("time", "dim2")
            )
            truncated_flux_t[:, :] = np.arange(1.0, 31.0).reshape(10, 3)
        yield test_file


def test_from_netcdf_truncates_flattened_2d_diagnostic(
    create_netcdf_file_with_truncated_2d_diagnostic,
):
    df = from_netcdf(
        create_netcdf_file_with_truncated_2d_diagnostic, "TruncatedFlux_t"
    ).data
    assert list(df.columns) == ["time", "TruncatedFlux_t"]
    # 30 flattened values truncated to the time length (10).
    np.testing.assert_array_equal(
        df["TruncatedFlux_t"].values,
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
    )
    assert len(df) == 10


# --- scalar / pad / unsupported shape cases --------------------------------
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
            grids_group.createVariable("time", np.float64, ("time",))[:] = np.arange(10)
            diagnostics_group.createVariable("scalar_st", np.float32).assignValue(3.5)
            diagnostics_group.createVariable(
                "PaddedSignal_st", np.float32, ("short_time",)
            )[:] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            diagnostics_group.createVariable(
                "ExactLengthSignal_t", np.float32, ("rows", "cols_exact")
            )[:, :] = np.arange(1.0, 11.0, dtype=np.float32).reshape(2, 5)
            diagnostics_group.createVariable(
                "PaddedMatrix_st", np.float32, ("rows", "cols_short")
            )[:, :] = np.array(
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
            )
            diagnostics_group.createVariable(
                "UnsupportedCube_t", np.float32, ("depth", "rows", "cols_short")
            )[:, :, :] = np.ones((2, 2, 3), dtype=np.float32)
            diagnostics_group.createVariable(
                "IgnoredMatrix", np.float32, ("rows", "cols_short")
            )[:, :] = np.ones((2, 3), dtype=np.float32)
        yield test_file


def test_from_netcdf_expands_scalar(create_netcdf_file_for_diagnostic_shape_cases):
    df = from_netcdf(create_netcdf_file_for_diagnostic_shape_cases, "scalar_st").data
    np.testing.assert_array_equal(df["time"].values, np.arange(10))
    np.testing.assert_array_equal(df["scalar_st"].values, np.full(10, 3.5))


def test_from_netcdf_pads_short_1d(create_netcdf_file_for_diagnostic_shape_cases):
    df = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, "PaddedSignal_st"
    ).data
    np.testing.assert_allclose(
        df["PaddedSignal_st"].values,
        np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )


def test_from_netcdf_flattens_exact_length_2d(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    df = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, "ExactLengthSignal_t"
    ).data
    np.testing.assert_array_equal(
        df["ExactLengthSignal_t"].values, np.arange(1.0, 11.0)
    )


def test_from_netcdf_pads_short_2d(create_netcdf_file_for_diagnostic_shape_cases):
    df = from_netcdf(
        create_netcdf_file_for_diagnostic_shape_cases, "PaddedMatrix_st"
    ).data
    np.testing.assert_allclose(
        df["PaddedMatrix_st"].values,
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )


def test_from_netcdf_ignores_unsupported_diagnostics(
    create_netcdf_file_for_diagnostic_shape_cases,
):
    # 3D (_t but ndim==3) and non-_t/_st-suffixed diagnostics are never extracted,
    # so requesting them raises "does not exist".
    for var in ("UnsupportedCube_t", "IgnoredMatrix"):
        with pytest.raises(ValueError, match="does not exist"):
            from_netcdf(create_netcdf_file_for_diagnostic_shape_cases, var)
