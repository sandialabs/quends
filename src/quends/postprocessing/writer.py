import json
from abc import ABC, abstractmethod
from typing import Any

from quends import DataStream
from quends.base.history import DataStreamHistory


# Abstract base class
class Writer(ABC):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def save(self, stream: DataStream) -> None:
        pass


# Handles JSON files specifically
class JsonWriter(Writer):
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
