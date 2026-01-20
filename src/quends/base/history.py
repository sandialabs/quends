from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence


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
