# quends/__init__.py

# Importing classes and functions from base module
from .base.data_stream import DataStream
from .base.ensemble import Ensemble
from .base.operations import DataStreamOperation
from .base.stationary import MakeDataStreamStationaryOperation
from .base.trim import (
    IQRTrimStrategy,
    MeanVariationTrimStrategy,
    NoiseThresholdTrimStrategy,
    QuantileTrimStrategy,
    RollingVarianceThresholdTrimStrategy,
    RollingVarianceTrimStrategy,
    SSSStartTrimStrategy,
    SelfConsistentTrimStrategy,
    StandardDeviationTrimStrategy,
    ThresholdTrimStrategy,
    TrimDataStreamOperation,
    TrimStrategy,
    build_trim_strategy,
)

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

# Importing classes from workflow module
from .workflow.batch_ensemble_workflow import BatchEnsembleWorkflow
from .workflow.ensemble_average_workflow import EnsembleAverageWorkflow
from .workflow.ensemble_statistics_workflow import EnsembleStatisticsWorkflow
from .workflow.robust_workflow import RobustWorkflow

# Optionally, you can define the __all__ variable to specify what is exported
# when using 'from quends import *'
__all__ = [
    # Core data containers
    "DataStream",
    "Ensemble",
    # Postprocessing
    "Exporter",
    "Plotter",
    # Preprocessing loaders
    "from_csv",
    "from_dict",
    "from_gx",
    "from_json",
    "from_netcdf",
    "from_numpy",
    # Workflow
    "RobustWorkflow",
    "EnsembleAverageWorkflow",
    "EnsembleStatisticsWorkflow",
    "BatchEnsembleWorkflow",
    # Trim — canonical entry point
    "build_trim_strategy",
    # Trim — strategy classes (concrete)
    "IQRTrimStrategy",
    "MeanVariationTrimStrategy",
    "NoiseThresholdTrimStrategy",
    "QuantileTrimStrategy",
    "RollingVarianceThresholdTrimStrategy",
    "SelfConsistentTrimStrategy",
    # Trim — aliases (backward compat)
    "StandardDeviationTrimStrategy",
    "ThresholdTrimStrategy",
    "RollingVarianceTrimStrategy",
    "SSSStartTrimStrategy",
    # Trim — operation + ABC
    "TrimStrategy",
    "TrimDataStreamOperation",
    # Other operations
    "DataStreamOperation",
    "MakeDataStreamStationaryOperation",
]
