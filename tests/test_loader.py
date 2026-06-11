import json

import pandas as pd
import pytest

from quends import DataStream
from quends.base.history import DataStreamHistory
from quends.postprocessing.loader import JsonLoader, Loader
from quends.postprocessing.writer import JsonWriter


def test_loader_is_abstract():
    with pytest.raises(TypeError):
        Loader("dummy.json")


def test_loader_concrete_subclass_load(tmp_path):
    file_path = tmp_path / "dummy.json"

    class DummyLoader(Loader):
        def load(self, filepath: str = None) -> DataStream:
            path = filepath or self.filepath
            self.loaded_from = path
            return DataStream(pd.DataFrame({"time": [0], "signal": [1.0]}))

    loader = DummyLoader(str(file_path))
    stream = loader.load()

    assert loader.filepath == str(file_path)
    assert loader.loaded_from == str(file_path)
    assert isinstance(stream, DataStream)
    assert stream.data.to_dict(orient="list") == {
        "time": [0],
        "signal": [1.0],
    }


def test_json_loader_reconstructs_stream(tmp_path):
    file_path = tmp_path / "stream.json"

    payload = {
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

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loader = JsonLoader(str(file_path))
    loaded_stream = loader.load()

    assert isinstance(loaded_stream, DataStream)
    assert isinstance(loaded_stream.history, DataStreamHistory)

    assert loaded_stream.data.to_dict(orient="list") == payload["data"]

    assert [
        {
            "operation_name": e.operation_name,
            "parameters": dict(e.parameters),
        }
        for e in loaded_stream.history.entries()
    ] == payload["metadata"]["history"]


def test_json_loader_loads_empty_history(tmp_path):
    file_path = tmp_path / "empty_history.json"
    payload = {
        "data": {"time": [0], "signal": [1.0]},
        "metadata": {"history": []},
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loader = JsonLoader(str(file_path))
    stream = loader.load()

    assert isinstance(stream.history, DataStreamHistory)
    assert tuple(stream.history.entries()) == ()


def test_json_loader_loads_empty_dataframe(tmp_path):
    file_path = tmp_path / "empty_data.json"
    payload = {
        "data": {"time": [], "signal": []},
        "metadata": {"history": []},
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loader = JsonLoader(str(file_path))
    stream = loader.load()

    assert stream.data.empty
    assert list(stream.data.columns) == ["time", "signal"]


def test_json_round_trip(tmp_path):
    file_path = tmp_path / "stream.json"

    stream = DataStream(pd.DataFrame({"time": [0], "signal": [1.0]}))

    writer = JsonWriter(str(file_path))
    writer.save(stream)

    loader = JsonLoader(str(file_path))
    loaded_stream = loader.load()

    assert loaded_stream.data.equals(stream.data)


def test_json_loader_missing_data_raises(tmp_path):
    file_path = tmp_path / "bad.json"
    payload = {"metadata": {"history": []}}

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loader = JsonLoader(str(file_path))

    with pytest.raises(KeyError):
        loader.load()
