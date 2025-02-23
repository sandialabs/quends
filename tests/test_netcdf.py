# Import statements
import pytest
import tempfile
import pandas as pd
import numpy as np
import netCDF4 as nc

# Special imports
from quends import DataStream
from quends import from_netcdf

# Fixture to create a NetCDF file in a temporary directory
# =============================================================================
@pytest.fixture
def create_netcdf_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.nc"
        
        with nc.Dataset(test_file, 'w', format='NETCDF4') as dataset:
            # Create dimensions
            dataset.createDimension('time', None) 

            # Create the "Grids" group
            grids_group = dataset.createGroup("Grids")

            # Create the "Diagnostics" group
            diagnostics_group = dataset.createGroup("Diagnostics")

            # Create variables in the "Grids" group
            time_var = grids_group.createVariable('time', np.float64, ('time',))
            heatflux_var = diagnostics_group.createVariable('HeatFlux_st', np.float32, ('time',))
            wg_var = diagnostics_group.createVariable('Wg_st', np.float32, ('time',))
            wphi_var = diagnostics_group.createVariable('Wphi_st', np.float32, ('time',))

            # Assign data to variables
            time_var[:] = np.arange(10)  # Time values from 0 to 9
            heatflux_var[:] = np.random.uniform(low=15.0, high=30.0, size=10)  # Random heat flux values
            wg_var[:] = np.random.uniform(low=0.0, high=10.0, size=10)  # Random wg values
            wphi_var[:] = np.random.uniform(low=0.0, high=360.0, size=10)  # Random wphi values

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
    assert hasattr(data_stream, 'df'), "DataStream does not have a 'data_frame' attribute."

    # Now you can proceed to check the contents of the DataFrame
    df = data_stream.df

    # Check if the DataFrame contains all expected columns
    expected_columns = ['time', 'HeatFlux_st', 'Wg_st', 'Wphi_st']
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."

    # Verify that the data values match what was written
    np.testing.assert_array_equal(df['time'].values, time_values, "Time values do not match.")
    np.testing.assert_array_equal(df['HeatFlux_st'].values, heatflux_values, "HeatFlux_st values do not match.")
    np.testing.assert_array_equal(df['Wg_st'].values, wg_values, "Wg_st values do not match.")
    np.testing.assert_array_equal(df['Wphi_st'].values, wphi_values, "Wphi_st values do not match.")

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries."

# Test 'from_netcdf' with nonexisting file
# =============================================================================
def test_from_netcdf_non_existent_file():
    # Define a path for a non-existent NetCDF file
    non_existent_file = "non_existent_file.nc"

    # Use pytest.raises to check for ValueError
    with pytest.raises(ValueError, match=f"Error: file {non_existent_file} does not exist."):
        from_netcdf(non_existent_file)
    
