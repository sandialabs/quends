from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

"""
Usage signature for the new trimming workflow::

    strategy = SomeTrimStrategy(kwargs...)
    trim_operation = TrimDataStreamOperation(strategy=strategy)
    data_stream = DataStream(data)
    trimmed_data_stream = trim_operation(data_stream)
"""

# This class represents an entry in the data stream history
# All of this could go into a separate file history.py for example
@dataclass(frozen=True)
class DataStreamHistoryEntry:
    """
    Immutable record describing a single operation performed on a data stream.
    """

    operation_name: str
    parameters: Mapping[str, Any]

# This class represents a history of operations performed on the data stream
class DataStreamHistory:
    """
    Ordered collection of operations applied to a DataStream.
    """

    # Initialize with optional list of entries
    def __init__(self, entries: Optional[Iterable[DataStreamHistoryEntry]] = None) -> None:
        self._entries: List[DataStreamHistoryEntry] = list(entries or [])

    def append(self, entry: DataStreamHistoryEntry) -> "DataStreamHistory":
        """
        Append ``entry`` to this history and return self for chaining.
        """

        self._entries.append(entry)
        return self

    def entries(self) -> Sequence[DataStreamHistoryEntry]:
        """
        Expose the ordered sequence of history entries.
        """

        return tuple(self._entries)
    
class DataStream:

    def __init__(self, data: Any, history: Optional[DataStreamHistory] = None) -> None:
        self._data = data
        self._history = history or DataStreamHistory()

    @property
    def data(self) -> Any:
        return self._data

    @property
    def history(self) -> DataStreamHistory:
        return self._history

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

# Now let's describe an abstract class for an "operation" to be performed on a data stream
# This could go into operations.py for example
class DataStreamOperation(ABC):

    # Initialize an operation with an optional name and optional kwargs
    # that describe extra history entries for this operation (e.g., the strategy
    # name in TrimDataStreamOperation below).
    def __init__(
        self,
        operation_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._operation_name = operation_name or self.__class__.__name__
        self._kwargs = dict(kwargs)

    @property
    def name(self) -> str:
        return self._operation_name

    def __call__(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        """
        Apply the concrete operation and record history metadata.
        """

        new_data_stream = self._apply(data_stream, **kwargs)
        history_entry = DataStreamHistoryEntry(
            operation_name=self.name,
            parameters={**self._kwargs, **kwargs},
        )
        new_data_stream.history.append(history_entry)
        return new_data_stream

    @abstractmethod
    def _apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        """
        Return a new DataStream after performing the underlying operation.
        """

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
