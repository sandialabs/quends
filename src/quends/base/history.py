from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

# This class represents an entry in the data stream history


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

    def copy(self) -> "DataStreamHistory":
        """Return an independent history object with the same immutable entries."""

        return DataStreamHistory(self._entries)

    def entries(self) -> Sequence[DataStreamHistoryEntry]:
        """
        Expose the ordered sequence of history entries.
        """

        return tuple(self._entries)

    # ------------------------------------------------------------------
    # Convenience accessors used by tests and legacy code that expects
    # list-of-dict access patterns.
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict:
        """Return entry *index* as a ``{"operation": …, "options": …}`` dict."""
        entry = self._entries[index]
        return {"operation": entry.operation_name, "options": dict(entry.parameters)}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DataStreamHistory):
            return self._entries == other._entries
        if isinstance(other, list):
            # Allow comparison with a list of {"operation": …, "options": …} dicts.
            if len(self._entries) != len(other):
                return False
            for entry, d in zip(self._entries, other):
                if not isinstance(d, dict):
                    return False
                if entry.operation_name != d.get("operation"):
                    return False
                if dict(entry.parameters) != d.get("options", {}):
                    return False
            return True
        return NotImplemented
