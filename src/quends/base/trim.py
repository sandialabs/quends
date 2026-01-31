from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.robust.scale import mad

from .data_stream import DataStream
from .operations import DataStreamOperation


# Abstract base class for a trim strategy
# This is the class that a potential third-party user could inherit
class TrimStrategy(ABC):
    """
    Abstract base class describing a trim strategy.
    Concrete strategies turn a data stream into a trimmed data stream.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        pass

    @abstractmethod
    def apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        """
        Return a trimmed representation of the given data stream.
        """


class StandardDeviationTrimStrategy(TrimStrategy):
    def __init__(
        self,
        batch_size: int = 10,
        start_time: float = 0.0,
        robust: bool = True,
        threshold: int = None,
    ):
        self.batch_size = batch_size
        self.start_time = start_time
        self.robust = robust
        self.threshold = threshold

    @property
    def method_name(self) -> str:
        return "std"

    def apply(
        self, data_stream: DataStream, column_name: str, **kwargs: Any
    ) -> DataStream:
        """
        Trim the DataStream to its steady-state portion.
        Strategy does the work, TrimDataStreamOperation handles history.
        """
        # Check for stationarity
        stationary_result = data_stream.is_stationary(column_name)
        is_stat = False
        if isinstance(stationary_result, dict):
            is_stat = stationary_result.get(column_name) is True

        if not is_stat:
            # Return empty result with error message
            empty_df = data_stream.data.iloc[0:0].copy()
            result = DataStream(empty_df, history=data_stream.history)
            result.message = (
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data."
            )
            return result

        # Preprocess
        data = data_stream.data[
            data_stream.data["time"] >= self.start_time
        ].reset_index(drop=True)
        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
            data = data.loc[non_zero_index:].reset_index(drop=True)

        # Steady-state detection
        steady_state_start_time = self._find_steady_state_std(
            data, column_name, window_size=self.batch_size, robust=self.robust
        )

        # Handle result based on whether steady state was found
        if steady_state_start_time is not None:
            trimmed_df = data_stream.data[  # Changed from self.data
                data_stream.data["time"] >= steady_state_start_time
            ][["time", column_name]].reset_index(drop=True)
            result = DataStream(trimmed_df, history=data_stream.history)
            kwargs["sss_start"] = steady_state_start_time
        else:
            empty_df = data_stream.data.iloc[0:0].copy()
            result = DataStream(empty_df, history=data_stream.history)
            kwargs["message"] = (
                f"Steady-state start time could not be determined for column '{column_name}'."
            )

        return result

    @staticmethod
    def _find_steady_state_std(
        data: pd.DataFrame, column_name: str, window_size: int = 10, robust: bool = True
    ):
        """
        Find steady state using standard deviation criteria.

        Identify the earliest time point when the signal remains within ±1/2/3σ proportions.
        """
        time_filtered = data["time"].values
        signal_filtered = data[column_name].values

        for i in range(len(signal_filtered) - window_size + 1):
            remaining_data = signal_filtered[i:]

            # Choose central tendency and scale based on robust flag
            if robust:
                central_value = np.median(remaining_data)
                scale_value = mad(remaining_data)
            else:
                central_value = np.mean(remaining_data)
                scale_value = np.std(remaining_data)

            # Check if data falls within expected proportions
            within_1 = np.mean(np.abs(remaining_data - central_value) <= scale_value)
            within_2 = np.mean(
                np.abs(remaining_data - central_value) <= 2 * scale_value
            )
            within_3 = np.mean(
                np.abs(remaining_data - central_value) <= 3 * scale_value
            )

            if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
                return time_filtered[i]

        return None


class ThresholdTrimStrategy(TrimStrategy):

    def apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        raise NotImplementedError("Implement threshold-specific trimming here")


class RollingVarianceTrimStrategy(TrimStrategy):

    def apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        raise NotImplementedError("Implement rolling variance trimming here")


class SSSStartTrimStrategy(TrimStrategy):

    def apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        raise NotImplementedError("Implement trim_sss_start() methodology")


class TrimDataStreamOperation(DataStreamOperation):
    """
    Operation that applies a TrimStrategy to a DataStream.
    """

    def __init__(self, strategy: TrimStrategy, operation_name: str = "trim") -> None:
        super().__init__(
            operation_name=operation_name,
            strategy=type(strategy).__name__,
        )
        self._strategy = strategy

    @property
    def strategy(self) -> TrimStrategy:
        return self._strategy

    # Override to skip base history
    def __call__(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        return self._apply(data_stream, **kwargs)

    def _apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        """Apply the strategy and add strategy-specific parameters to kwargs."""
        column_name = kwargs.get("column_name")

        options = {
            "column_name": column_name,
            "batch_size": self._strategy.batch_size,
            "start_time": self._strategy.start_time,
            "method": self._strategy.method_name,
            "threshold": 4,
            "robust": self._strategy.robust,
        }

        # Apply strategy
        result = self._strategy.apply(
            data_stream,
            **options,
        )

        # Strategy may inject message / sss_start
        if hasattr(result, "message"):
            options["message"] = result.message
        if "sss_start" in kwargs:
            options["sss_start"] = kwargs["sss_start"]

        result._history = [
            {"operation": "is_stationary", "options": {"columns": column_name}},
            {"operation": "trim", "options": options},
        ]

        # Call the strategy
        return result
