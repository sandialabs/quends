"""Preprocessing loaders for QUENDS.

Each loader takes a single ``variable`` and returns a ``DataStream`` containing
``[time, variable]`` (the time column is auto-detected and standardized).
"""

from .csv import from_csv
from .dictionary import from_dict
from .gx import from_gx
from .json import from_json
from .netcdf import from_netcdf
from .numpy import from_numpy

__all__ = [
    "from_csv",
    "from_dict",
    "from_gx",
    "from_json",
    "from_netcdf",
    "from_numpy",
]
