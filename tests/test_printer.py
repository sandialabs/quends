import json

import pandas as pd
import pytest

from quends import DataStream, QuantileTrimStrategy, TrimDataStreamOperation
from quends.base.history import DataStreamHistory, DataStreamHistoryEntry
from quends.postprocessing.printer import JsonWriter

pytest_plugins = ("tests._shared",)


def test_json_writer_round_trip(tmp_path):
    file_path = tmp_path / "stream.json"
    stream = DataStream(
        pd.DataFrame(
            {
                "time": [0, 1, 2],
                "signal": [1.5, 2.5, 3.5],
            }
        ),
        history=DataStreamHistory(
            [
                DataStreamHistoryEntry(
                    operation_name="trim",
                    parameters={"column_name": "signal", "window_size": 3},
                )
            ]
        ),
    )

    writer = JsonWriter(str(file_path))
    writer.save(stream)

    with open(file_path, encoding="utf-8") as f:
        saved_payload = json.load(f)

    assert saved_payload == {
        "data": {
            "time": [0, 1, 2],
            "signal": [1.5, 2.5, 3.5],
        },
        "metadata": {
            "history": [
                {
                    "operation_name": "trim",
                    "parameters": {"column_name": "signal", "window_size": 3},
                }
            ]
        },
    }

    loaded_stream = writer.load()

    assert isinstance(loaded_stream, DataStream)
    pd.testing.assert_frame_equal(loaded_stream.data, stream.data)
    assert loaded_stream.history.entries() == stream.history.entries()


def test_json_writer_round_trip_multiple_history_entries(tmp_path):
    file_path = tmp_path / "stream_multiple_history.json"
    stream = DataStream(
        pd.DataFrame({"time": [0, 1], "signal": [4.0, 5.0]}),
        history=DataStreamHistory(
            [
                DataStreamHistoryEntry(
                    operation_name="normalize",
                    parameters={"column_name": "signal"},
                ),
                DataStreamHistoryEntry(
                    operation_name="smooth",
                    parameters={"window_size": 5, "method": "moving_average"},
                ),
            ]
        ),
    )

    writer = JsonWriter(str(file_path))
    writer.save(stream)
    loaded_stream = writer.load()

    assert isinstance(loaded_stream.history, DataStreamHistory)
    assert loaded_stream.history.entries() == stream.history.entries()
    pd.testing.assert_frame_equal(loaded_stream.data, stream.data)


def test_json_writer_round_trip_real_trim_history(tmp_path, trim_data):
    file_path = tmp_path / "trimmed_stream.json"
    stream = DataStream(trim_data)
    trim_op = TrimDataStreamOperation(
        strategy=QuantileTrimStrategy(window_size=1, start_time=3.0, robust=True)
    )

    trimmed_stream = trim_op(stream, column_name="A")
    writer = JsonWriter(str(file_path))
    writer.save(trimmed_stream)

    with open(file_path, encoding="utf-8") as f:
        saved_payload = json.load(f)

    loaded_stream = writer.load()

    assert saved_payload["metadata"]["history"] == trimmed_stream.history
    assert isinstance(loaded_stream.history, list)
    assert loaded_stream.history == trimmed_stream.history
    assert loaded_stream.data.empty
    assert list(loaded_stream.data.columns) == list(trimmed_stream.data.columns)
    assert loaded_stream.data.to_dict(orient="list") == trimmed_stream.data.to_dict(
        orient="list"
    )


def test_json_writer_overwrite(tmp_path):
    file_path = tmp_path / "overwrite.json"
    writer = JsonWriter(str(file_path))

    writer.save(DataStream(pd.DataFrame({"time": [0], "signal": [1.0]})))
    writer.save(DataStream(pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0]})))

    loaded = writer.load()

    assert loaded.data.to_dict(orient="list") == {
        "time": [0, 1],
        "signal": [1.0, 2.0],
    }


def test_json_writer_load_missing_file_raises(tmp_path):
    writer = JsonWriter(str(tmp_path / "does_not_exist.json"))
    with pytest.raises(FileNotFoundError):
        writer.load()
