from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.robust.scale import mad

from .data_stream import DataStream
from .operations import DataStreamOperation


class TrimStrategy(ABC):
    """
    Abstract base class describing a trim strategy.
    """

    def __init__(self, window_size: int = 10, start_time: float = 0.0, **kwargs):
        self.window_size = window_size
        self.start_time = start_time
        # Store any extra kwargs (like threshold, robust, etc.)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        pass

    def apply(
        self, data_stream: DataStream, column_name: str, **kwargs: Any
    ) -> DataStream:
        """
        Template method that defines the trimming workflow.
        """
        # Check stationarity
        if not self._check_stationary(data_stream, column_name):
            return self._create_error_result(
                data_stream,
                kwargs,
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data.",
            )

        # Preprocess
        data = self._preprocess(data_stream, column_name)

        # Detect steady state (implemented by subclasses)
        steady_state_start_time = self._detection_method(data, column_name)

        # Handle result
        if steady_state_start_time is not None:
            return self._create_success_result(
                data_stream, column_name, steady_state_start_time, kwargs
            )
        else:
            return self._create_error_result(
                data_stream,
                kwargs,
                f"Steady-state start time could not be determined for column '{column_name}'.",
            )

    def _check_stationary(self, data_stream: DataStream, column_name: str) -> bool:
        """Check if the column is stationary."""
        stationary_result = data_stream.is_stationary(column_name)
        if isinstance(stationary_result, dict):
            return stationary_result.get(column_name) is True
        return False

    def _preprocess(self, data_stream: DataStream, column_name: str) -> pd.DataFrame:
        """Preprocess the data"""
        data = data_stream.data[
            data_stream.data["time"] >= self.start_time
        ].reset_index(drop=True)

        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
            data = data.loc[non_zero_index:].reset_index(drop=True)

        return data

    def _create_success_result(
        self,
        data_stream: DataStream,
        column_name: str,
        steady_state_start_time: float,
        kwargs: dict,
    ) -> DataStream:
        """Create a successful trimmed result."""
        trimmed_df = data_stream.data[
            data_stream.data["time"] >= steady_state_start_time
        ][["time", column_name]].reset_index(drop=True)

        result = DataStream(trimmed_df, history=data_stream.history)
        kwargs["sss_start"] = steady_state_start_time
        return result

    def _create_error_result(
        self, data_stream: DataStream, kwargs: dict, message: str
    ) -> DataStream:
        """Create an empty result with error message."""
        empty_df = data_stream.data.iloc[0:0].copy()
        result = DataStream(empty_df, history=data_stream.history)
        result.message = message
        kwargs["message"] = message
        return result

    @abstractmethod
    def _detection_method(
        self, data: pd.DataFrame, column_name: str
    ) -> Optional[float]:
        """
        Detect the steady-state start time.
        Must be implemented by subclasses.

        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed data
        column_name : str
            Name of the column to analyze

        Returns
        -------
        float or None
            The time at which steady state begins, or None if not found.
        """
        pass


class StandardDeviationTrimStrategy(TrimStrategy):
    """Trim based on sliding standard deviation criteria."""

    def __init__(
        self,
        window_size: int = 10,
        start_time: float = 0.0,
        robust: bool = True,
    ):
        super().__init__(
            window_size=window_size,
            start_time=start_time,
            robust=robust,
        )

    @property
    def method_name(self) -> str:
        return "std"

    def _detection_method(
        self,
        data: pd.DataFrame,
        column_name: str,
    ) -> Optional[float]:
        """
        Identify the earliest time point when the signal remains within ±1/2/3σ proportions.

        Parameters
        ----------
        data : DataFrame
            Subset of the original df (must include 'time' and signal column).
        column_name : str
        window_size : int
            Number of samples to evaluate the steady-state criteria.
        robust : bool
            If True, use median and MAD; else mean and std.

        Returns
        -------
        """

        window_size = self.window_size
        robust = self.robust

        time_filtered = data["time"].values
        signal_filtered = data[column_name].values

        if len(signal_filtered) < window_size:
            return None

        for i in range(len(signal_filtered) - window_size + 1):
            remaining_data = signal_filtered[i:]

            if robust:
                central_value = np.median(remaining_data)
                scale_value = mad(remaining_data)
            else:
                central_value = np.mean(remaining_data)
                scale_value = np.std(remaining_data)

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
    """Trim using rolling standard deviation on normalized data."""

    def __init__(
        self,
        window_size: int = 10,
        start_time: float = 0.0,
        threshold: Optional[float] = None,
        robust: bool = True,
    ):
        super().__init__(
            window_size=window_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )

    def apply(self, data_stream: DataStream, column_name: str, **kwargs):
        # Stationarity check
        if not self._check_stationary(data_stream, column_name):
            return self._create_error_result(
                data_stream,
                kwargs,
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data.",
            )

        # Now threshold validation
        if self.threshold is None:
            return self._create_error_result(
                data_stream,
                kwargs,
                "Threshold must be specified for the 'threshold' trim strategy.",
            )

        return super().apply(data_stream, column_name, **kwargs)

    @property
    def method_name(self) -> str:
        return "threshold"

    def _detection_method(
        self,
        data: pd.DataFrame,
        column_name: str,
    ) -> Optional[float]:
        """
        Use rolling standard deviation on normalized data to detect steady-state.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Std threshold under which to mark steady-state.

        Returns
        -------
        float or None
        """
        window_size = self.window_size
        threshold = self.threshold

        normalized_data = DataStream.normalize_data(data.copy())
        time_series = normalized_data[["time", column_name]]

        if len(time_series) < window_size:
            return None

        rolling_std = time_series[column_name].rolling(window=window_size).std()
        common_idx = time_series.index.intersection(rolling_std.index)
        steady_state = time_series.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]

        if not steady_state.empty:
            return steady_state.iloc[0]

        return None


class RollingVarianceTrimStrategy(TrimStrategy):
    """Detect steady-state when rolling variance falls below threshold."""

    def __init__(
        self,
        batch_size: int = 10,
        start_time: float = 0.0,
        robust: bool = True,
    ):
        super().__init__(batch_size=batch_size, start_time=start_time, robust=robust)

    @property
    def method_name(self) -> str:
        return "rolling_variance"

    def _detection_method(
        self, data: pd.DataFrame, column_name: str
    ) -> Optional[float]:
        pass


class SSSStartTrimStrategy(TrimStrategy):
    """Trim using Statistical Steady State detection."""

    def __init__(self, workflow):
        # SSSStart doesn't use batch_size/start_time the same way
        super().__init__(batch_size=0, start_time=0.0, workflow=workflow)

    @property
    def method_name(self) -> str:
        return "sss_start"

    def _detection_method(
        self, data: pd.DataFrame, column_name: str
    ) -> Optional[float]:
        """Detect using SSS methodology."""
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
        """Apply the strategy and build history."""
        column_name = kwargs.get("column_name")

        # Build base parameters that all strategies share
        options = {
            "column_name": column_name,
            "batch_size": self._strategy.window_size,
            "start_time": self._strategy.start_time,
            "method": self._strategy.method_name,
        }

        # Add strategy-specific parameters
        if hasattr(self._strategy, "robust"):
            options["robust"] = self._strategy.robust
        if hasattr(self._strategy, "threshold"):
            options["threshold"] = self._strategy.threshold

        # Apply strategy
        result = self._strategy.apply(data_stream, **kwargs)

        # Strategy may inject message / sss_start into kwargs
        if hasattr(result, "message"):
            options["message"] = result.message
        if "sss_start" in kwargs:
            options["sss_start"] = kwargs["sss_start"]

        # Build history as a dictionary
        result._history = [
            {"operation": "is_stationary", "options": {"columns": column_name}},
            {"operation": "trim", "options": options},
        ]

        return result
