# Import statements
import pytest
import tempfile
import pandas as pd
import numpy as np
import netCDF4 as nc

# Special imports
from quends import DataStream
from quends import from_gx


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

# Fixture to create an empty NetCDF file in a temporary directory
# =============================================================================
@pytest.fixture
def create_empty_netcdf_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_empty_file = f"{temp_dir}/test_empty.nc"
        
        with nc.Dataset(test_empty_file, 'w', format='NETCDF4') as dataset:
            # Create dimensions
            dataset.createDimension('time', None) 

            # Create the "Grids" group
            grids_group = dataset.createGroup("Grids")

            # Create the "Diagnostics" group
            diagnostics_group = dataset.createGroup("Diagnostics")

            # Create variables in the "Grids" group
            time_var = grids_group.createVariable('time', np.float64, ('time',))

        yield test_empty_file  # Yield the path to the created NetCDF file

# Fixture to create a CSV file in a temporary directory
# =============================================================================
@pytest.fixture
def create_csv_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.csv"
        
        # Create sample data
        data = {
            'time': range(10),
            'HeatFlux_st': [20.5, 21.0, 19.5, 22.0, 23.5, 24.0, 25.0, 26.5, 27.0, 28.0],
            'Wg_st': [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
            'Wphi_st': [180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
        }
        df = pd.DataFrame(data)
        df.to_csv(test_file, index=False)  # Save DataFrame to CSV without the index
        
        yield test_file  # Yield the path to the created CSV file

# Test 'from_gx' with no variables assigned
# =============================================================================
def test_without_variables(create_netcdf_file):
    test_file, time_values, heatflux_values, wg_values, wphi_values = create_netcdf_file  # Get the filename and original values

    data_stream = from_gx(test_file)
    
     # Check if it is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object"
    print(f"Returning object type: {type(data_stream)}")

    # Check if DataStream has DataFrame
    assert hasattr(data_stream, 'df'), "DataStream does not have a 'df'"
    
    # Check if the DataFrame contains the expected data
    df = data_stream.df
    
    # Check if the DataFrame contains all expected columns
    expected_columns = ['time', 'HeatFlux_st', 'Wg_st', 'Wphi_st']
    for column in expected_columns:
        assert column in df.columns, f"DataFrame should contain '{column}' column."
        
    # Validate the contents of the DataFrame
    assert len(df) == 10  # Ensure there are 10 entries

    # Verify that the data values match what was written
    np.testing.assert_array_equal(df['time'].values, time_values, "Time values do not match.")
    np.testing.assert_array_equal(df['HeatFlux_st'].values, heatflux_values, "HeatFlux_st values do not match.")
    np.testing.assert_array_equal(df['Wg_st'].values, wg_values, "Wg_st values do not match.")
    np.testing.assert_array_equal(df['Wphi_st'].values, wphi_values, "Wphi_st values do not match.")


# Test 'from_gx' with variables assigned
# =============================================================================
def test_with_variables(create_netcdf_file):
    test_file, time_values, heatflux_values, wg_values, wphi_values = create_netcdf_file  # Get the filename and original values

    data_stream = from_gx(test_file, variables=["time", "HeatFlux_st"])
    
     # Check if it is a DataStream
    assert isinstance(data_stream, DataStream), "Expected a DataStream object"
    print(f"Returning object type: {type(data_stream)}")

    # Check if DataStream has DataFrame
    assert hasattr(data_stream, 'df'), "DataStream does not have a 'df'"
    
    # Check if the DataFrame contains the expected data
    df = data_stream.df
    assert 'time' in df.columns, "DataFrame should contain 'time' column."
    assert 'HeatFlux_st' in df.columns, "DataFrame should contain 'HeatFlux_st' column."
    
     # Ensure that unwanted variables are not present
    assert 'Wg_st' not in df.columns, "DataFrame should not contain 'Wg_st' column."
    assert 'Wphi_st' not in df.columns, "DataFrame should not contain 'Wphi_st' column."

    # Validate the contents of the DataFrame
    assert len(df) == 10, "DataFrame should have 10 entries"

    # Verify that the data values match what was written
    np.testing.assert_array_equal(df['time'].values, time_values, "Time values do not match.")
    np.testing.assert_array_equal(df['HeatFlux_st'].values, heatflux_values, "HeatFlux_st values do not match.")

# Test loading from a non-existent file
# =============================================================================
def test_invalid_file():
    with pytest.raises(ValueError, match="Error: file .* does not exist."):
        from_gx("non_existent_file.nc")

# Test loading from an unsupported file format
def test_unsupported_file_format():
    """Test that from_gx raises a ValueError for unsupported file formats."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b"This is a test file with an unsupported format.")
        temp_file.close()

        with pytest.raises(ValueError, match="Unsupported file format. Please provide a .nc or .csv file."):
            from_gx(temp_file.name)

# Test loading from a CSV file without specifying variables
# =============================================================================
def test_csv_without_variables(create_csv_file):
    test_file = create_csv_file
    data_stream = from_gx(test_file)
    assert isinstance(data_stream, DataStream)
    df = data_stream.df
    expected_columns = ['time', 'HeatFlux_st', 'Wg_st', 'Wphi_st']
    for column in expected_columns:
        assert column in df.columns
    assert len(df) == 10

# Test loading specific variables from a CSV file
# =============================================================================
def test_csv_with_specific_variables(create_csv_file):
    test_file = create_csv_file
    data_stream = from_gx(test_file, variables=['time', 'HeatFlux_st'])
    assert isinstance(data_stream, DataStream)
    df = data_stream.df
    assert 'time' in df.columns
    assert 'HeatFlux_st' in df.columns
    assert 'Wg_st' not in df.columns
    assert 'Wphi_st' not in df.columns
    assert len(df) == 10
