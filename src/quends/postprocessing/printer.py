import json
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quends import DataStream
from quends.base.history import DataStreamHistory, DataStreamHistoryEntry


# Abstract base class
class Printer(ABC):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def save(self, stream: DataStream) -> None:
        pass

    @abstractmethod
    def load(self) -> DataStream:
        pass


# Handles JSON files specifically
class JsonWriter(Printer):
    def __init__(self, filepath: str, indent: int = 2):
        super().__init__(filepath)
        self.indent = indent

    # Helper to convert the DataStreamHistory into plain list of dicitonaries
    # JSON can't store Python objects directly
    @staticmethod
    def _serialize_history(history: Any) -> Any:
        if isinstance(history, DataStreamHistory):
            return [
                {
                    "operation_name": entry.operation_name,
                    "parameters": dict(entry.parameters),
                }
                for entry in history.entries()
            ]
        return history

    # Helper to convert the list of dictionaries back into a DataStreamHistory object
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

    def save(self, stream: DataStream) -> None:
        """
        Save the DataStream to a JSON file, including its history."""
        payload = {
            "data": stream.data.to_dict(orient="list"),
            "metadata": {"history": self._serialize_history(stream.history)},
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=self.indent)
        print(f"[JsonWriter] saved -> {self.filepath}")

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
