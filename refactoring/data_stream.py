from typing import Any, Optional

from .history import DataStreamHistory

"""
Usage signature for the new trimming workflow::

    strategy = SomeTrimStrategy(kwargs...)
    trim_operation = TrimDataStreamOperation(strategy=strategy)
    data_stream = DataStream(data)
    trimmed_data_stream = trim_operation(data_stream)
"""


# Goes into datastream
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
