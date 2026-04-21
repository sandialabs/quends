import json

import pandas as pd
import pytest

from quends import DataStream, from_json


@pytest.fixture
def create_json_file(tmp_path):
    test_file = tmp_path / "test_data.json"
    test_data = {
        "time": [0, 1, 2, 3, 4],
        "HeatFlux_st": [15.0, 16.5, 18.0, 19.5, 21.0],
        "Wg_st": [0.5, 1.0, 1.5, 2.0, 2.5],
        "Wphi_st": [0.0, 45.0, 90.0, 135.0, 180.0],
    }
    test_file.write_text(json.dumps(test_data), encoding="utf-8")
    return str(test_file)


def test_from_json_loads_single_requested_variable(create_json_file):
    data_stream = from_json(create_json_file, variable="HeatFlux_st")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data,
        pd.DataFrame({"HeatFlux_st": [15.0, 16.5, 18.0, 19.5, 21.0]}),
    )


def test_from_json_writer_payload_raises_for_missing_variable_shape(tmp_path):
    test_file = tmp_path / "writer_payload.json"
    test_file.write_text(
        json.dumps(
            {
                "data": {
                    "time": [0, 1, 2],
                    "HeatFlux_st": [10.0, 20.0, 30.0],
                },
                "metadata": {"history": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="Error: variable 'HeatFlux_st' does not exist in file"
    ):
        from_json(str(test_file), variable="HeatFlux_st")


def test_from_json_non_existent_file():
    non_existent_file = "non_existent_file.json"

    with pytest.raises(
        ValueError, match=f"Error: file {non_existent_file} does not exist."
    ):
        from_json(non_existent_file, variable="HeatFlux_st")


def test_from_json_invalid_file(tmp_path):
    invalid_json_file = tmp_path / "invalid_data.json"
    invalid_json_file.write_text("This is not a valid JSON.", encoding="utf-8")

    with pytest.raises(ValueError):
        from_json(str(invalid_json_file), variable="HeatFlux_st")


def test_from_json_missing_variable_raises(create_json_file):
    with pytest.raises(
        ValueError, match="Error: variable 'missing' does not exist in file"
    ):
        from_json(create_json_file, variable="missing")
