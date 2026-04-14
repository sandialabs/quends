# Helper to convert the list of dictionaries back into a DataStreamHistory object
import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quends import DataStream
from quends.base.history import DataStreamHistory, DataStreamHistoryEntry


# Abstract base class for loaders
class Loader(ABC):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def load(self, filepath: str = None) -> DataStream:
        pass


# Concrete loader for JSON files
class JsonLoader(Loader):
    @staticmethod
    def _deserialize_history(history_payload: Any) -> Any:
        if isinstance(history_payload, list) and all(
            isinstance(entry, dict)
            and "operation_name" in entry
            and "parameters" in entry
            for entry in history_payload
        ):
            return DataStreamHistory(
                [
                    DataStreamHistoryEntry(
                        operation_name=entry["operation_name"],
                        parameters=entry.get("parameters", {}),
                    )
                    for entry in history_payload
                ]
            )
        return history_payload

    def load(self, filepath: str = None) -> DataStream:
        """
        Load a DataStream from a JSON file, reconstructing its history.
        """
        path_to_load = filepath or self.filepath
        with open(path_to_load, encoding="utf-8") as f:
            payload = json.load(f)
        history = self._deserialize_history(
            payload.get("metadata", {}).get("history", [])
        )
        return DataStream(pd.DataFrame(payload["data"]), history=history)
