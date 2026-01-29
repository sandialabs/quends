from typing import Any, Optional

import numpy as np

from ..src.quends.base.history import DataStreamHistory

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

    def head(self, n: int = 5) -> Any:
        return self.data.head(n)

    def __len__(self) -> int:
        return len(self.data)

    # --------- Statistical summaries (unchanged except metadata) --------
    def _mean(self, column_name=None, method="non-overlapping", window_size=None):
        """
        Compute block or sliding window means for each column.

        Private helper for compute_statistics and confidence intervals.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.df[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            est_win = self._estimate_window(col, column_data, window_size)
            proc_data = self._process_column(column_data, est_win, method)
            results[col] = {
                "mean": float(np.mean(proc_data)),
                "window_size": int(est_win),
            }
        return results
