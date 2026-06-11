import json

import pandas as pd
import pytest

from quends import DataStream
from quends.base.history import DataStreamHistory, DataStreamHistoryEntry
from quends.postprocessing.writer import JsonWriter, Writer


def test_writer_is_abstract():
    with pytest.raises(TypeError):
        Writer("dummy.json")


def test_writer_concrete_subclass_save(tmp_path):
    file_path = tmp_path / "dummy.json"

    class DummyWriter(Writer):
        def save(self, stream: DataStream) -> None:
            self.saved_stream = stream

    writer = DummyWriter(str(file_path))
    stream = DataStream(pd.DataFrame({"time": [0], "signal": [1.0]}))

    writer.save(stream)

    assert writer.filepath == str(file_path)
    assert writer.saved_stream is stream


def test_json_writer_saves_correct_payload(tmp_path):
    file_path = tmp_path / "stream.json"

    stream = DataStream(
        pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]}),
        history=DataStreamHistory(
            [
                DataStreamHistoryEntry(
                    operation_name="trim",
                    parameters={"column_name": "signal"},
                )
            ]
        ),
    )

    writer = JsonWriter(str(file_path))
    writer.save(stream)

    with open(file_path, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload == {
        "data": {
            "time": [0, 1],
            "signal": [1.0, 2.0],
        },
        "metadata": {
            "history": [
                {
                    "operation_name": "trim",
                    "parameters": {"column_name": "signal"},
                }
            ]
        },
    }


def test_json_writer_saves_empty_history(tmp_path):
    file_path = tmp_path / "empty_history.json"
    stream = DataStream(pd.DataFrame({"time": [0], "signal": [1.0]}))

    writer = JsonWriter(str(file_path))
    writer.save(stream)

    with open(file_path, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["metadata"]["history"] == []


def test_json_writer_saves_empty_dataframe(tmp_path):
    file_path = tmp_path / "empty_data.json"
    stream = DataStream(pd.DataFrame(columns=["time", "signal"]))

    writer = JsonWriter(str(file_path))
    writer.save(stream)

    with open(file_path, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["data"] == {"time": [], "signal": []}


def test_json_writer_missing_data_raises(tmp_path):
    file_path = tmp_path / "bad.json"
    writer = JsonWriter(str(file_path))

    class BadStream:
        history = []

    with pytest.raises(AttributeError):
        writer.save(BadStream())
