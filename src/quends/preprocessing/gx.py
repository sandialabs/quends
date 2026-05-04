# Import statements

# Special imports
from .csv import from_csv
from .netcdf import from_netcdf


# Function to load an ensemble of GX data streams
def from_gx(file, variable):
    """
    Load a single variable from a GX output file (.nc or .csv)
    """

    # Select variables
    if not variable:
        print("No variable specified.\n")
        print(
            "Please specify a variable to load only that variable (e.g. variable='temperature')."
        )
        return

    # Load data stream and determine file type
    if file.endswith(".nc"):
        loader = from_netcdf
    elif file.endswith(".csv"):
        loader = from_csv
    else:
        raise ValueError("Unsupported file format. Please provide a .nc or .csv file.")

    # load each variable separately
    return loader(file, variable=variable)
