import numpy as np
import pandas as pd

from quends.base.data_stream import DataStream


class RobustWorkflow:
    """
    Set of functions to analyze DataStreams in a robust way. 

    This class can handle data streams with a lot of noise and where stationarity or the start
    of steady statistical state (SSS) can be hard to assess. It uses DataStream methods for statistical 
    analysis but adds alternative tools for stationarity assessment and start of SSS detection.

    Core features include:
    - Stationarity assessment that progressively shortens the DataStream to see if the tail
      end of the DataStream is stationary.
    - Start of SSS detection that uses a robust approach based on the smoothed mean of the DataStream.
    - Methods that return "ball park" statistics if the DataStream is not stationary, 
      or if there is no SSS segment found.

    Attributes
    ----------
    _drop_fraction: fraction of data to drop from the start of the DataStream to see if the shortened
        DataStream is stationary.

    """

    def __init__(self, drop_fraction=0.25):
        """
        Initialize a workflow and its hyperparameters

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain a 'time' column and one or more signal columns.
        _history : list of dict, optional
            Existing operation history to inherit.  If None, starts empty.
        """
        self._drop_fraction = drop_fraction
