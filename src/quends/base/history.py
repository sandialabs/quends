from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def to_native_types(obj):
    """
    Recursively convert NumPy scalar and array types in nested structures to native Python types.

    This function walks through dictionaries, lists, tuples, NumPy scalars, and arrays,
    converting them into Python built-ins:

    - NumPy scalar → Python int or float
    - NumPy array  → Python list (recursively)

    Parameters
    ----------
    obj : any
        The object to convert. Supported container types are dict, list, tuple,
        NumPy ndarray/scalar. Other types are returned unchanged.

    Returns
    -------
    any
        A new object mirroring the input structure but with all NumPy data types replaced
        by their native Python equivalents.
    """
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        return t([to_native_types(v) for v in obj])
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    else:
        return obj


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
    def __init__(
        self, entries: Optional[Iterable[DataStreamHistoryEntry]] = None
    ) -> None:
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

    def deduplicate(self) -> "DataStreamHistory":
        """
        Create a new DataStreamHistory keeping only the most recent occurance
        of each operation.

        Scans the history of operations (each represented as a dict with at least an 'operation' key)
        from end to start, retaining only the last entry for each unique operation name while preserving
        the overall order of those final occurrences.

        Returns
        -------
        DataStreamHistory
            a new history object with deduplicated entries
        """

        seen = set()
        out = []
        # Reverse to keep last call
        for entry in reversed(self._entries):
            if entry.operation_name not in seen:
                out.append(entry)
                seen.add(entry.operation_name)

        # Reverse again to restore original order
        return DataStreamHistory(reversed(out))
