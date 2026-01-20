from abc import ABC, abstractmethod
from typing import Any

from .data_stream import DataStream
from .operations import DataStreamOperation

"""
Usage signature for the new trimming workflow::

    strategy = SomeTrimStrategy(kwargs...)
    trim_operation = TrimDataStreamOperation(strategy=strategy)
    data_stream = DataStream(data)
    trimmed_data_stream = trim_operation(data_stream)
"""

# Goes into datastream
# class DataStream:

#     def __init__(self, data: Any, history: Optional[DataStreamHistory] = None) -> None:
#         self._data = data
#         self._history = history or DataStreamHistory()

#     @property
#     def data(self) -> Any:
#         return self._data

#     @property
#     def history(self) -> DataStreamHistory:
#         return self._history

# Now let's describe different trim strategies


# Abstract base class for a trim strategy
# This is the class that a potential third-party user could inherit
class TrimStrategy(ABC):
    """
    Abstract base class describing a trim strategy.
    Concrete strategies turn a data stream into a trimmed data stream.
    """

    @abstractmethod
    def apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        """
        Return a trimmed representation of the given data stream.
        """


class StandardDeviationTrimStrategy(TrimStrategy):

    def apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        raise NotImplementedError("Implement STD-specific trimming here")


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

    def _apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        return self._strategy.apply(data_stream, **kwargs)
