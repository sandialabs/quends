# quends/__init__.py

# Importing classes and functions from base module
from .base.data_stream import DataStream
from .base.ensemble import Ensemble

# Importing classes and functions from postprocessing module
from .postprocessing.exporter import Exporter
from .postprocessing.plotter import Plotter

# Importing functions from preprocessing module
from .preprocessing.csv import from_csv
from .preprocessing.dictionary import from_dict
from .preprocessing.gx import from_gx
from .preprocessing.json import from_json
from .preprocessing.netcdf import from_netcdf
from .preprocessing.numpy import from_numpy

# Optionally, you can define the __all__ variable to specify what is exported when using 'from quends import *'
__all__ = [
    "DataStream",
    "Ensemble",
    "Exporter",
    "Plotter",
    "from_csv",
    "from_dict",
    "from_gx",
    "from_json",
    "from_netcdf",
    "from_numpy",
]
