# Import statements

# Special imports
from .csv import from_csv
from .netcdf import from_netcdf

# Default variables to load for GX
DEFAULT_VARIABLES = ["time", "HeatFlux_st", "Wg_st", "Wphi_st"]


# Function to load an ensemble of GX data streams
def from_gx(file, variables=None):
    """
    Load a data stream from GX outputs.
    """

    # Select variables
    if not variables:
        variables = DEFAULT_VARIABLES

    # Load data stream and determine file type
    if file.endswith("nc"):
        loader = from_netcdf
    elif file.endswith(".csv"):
        loader = from_csv
    else:
        raise ValueError("Unsupported file format. Please provide a .nc or .csv file.")

    # load each variable separately
    return {variable: loader(file, variable=variable) for variable in variables}
