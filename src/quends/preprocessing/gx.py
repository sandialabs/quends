# Import statements

# Special imports
from .csv import from_csv
from .netcdf import from_netcdf


# Function to load a single GX data-stream variable
def from_gx(file, variable):
    """
    Load a single variable from a GX output file (.nc or .csv).

    Args:
        file (str): Path to the GX output file (``.nc`` or ``.csv``).
        variable (str): The variable name to load (e.g. ``"HeatFlux_st"``).

    Returns:
        DataStream: A DataStream containing ``[time, variable]``.

    Raises:
        ValueError: If no variable is specified or the file format is unsupported.
    """
    # A variable must be specified.
    if not variable:
        raise ValueError(
            "No variable specified. Please specify a variable to load "
            "(e.g. variable='HeatFlux_st')."
        )

    # Load data stream and determine file type.
    if file.endswith(".nc"):
        return from_netcdf(file, variable)
    elif file.endswith(".csv"):
        return from_csv(file, variable)
    else:
        raise ValueError("Unsupported file format. Please provide a .nc or .csv file.")
