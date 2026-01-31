# Now let's describe an abstract class for an "operation" to be performed on a data stream
# This could go into operations.py for example

from abc import ABC, abstractmethod
from typing import Any, Optional

from .data_stream import DataStream
from .history import DataStreamHistoryEntry


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
