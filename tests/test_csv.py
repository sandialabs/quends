import tempfile

import pandas as pd
import pytest

from quends import DataStream, from_csv


@pytest.fixture
def create_csv_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = f"{temp_dir}/test.csv"
        df = pd.DataFrame(
            {
                "time": range(10),
                "HeatFlux_st": [
                    20.5,
                    21.0,
                    19.5,
                    22.0,
                    23.5,
                    24.0,
                    25.0,
                    26.5,
                    27.0,
                    28.0,
                ],
                "Wg_st": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                "Wphi_st": [180, 190, 200, 210, 220, 230, 240, 250, 260, 270],
            }
        )
        df.to_csv(test_file, index=False)
        yield test_file


def test_from_csv_loads_single_requested_variable(create_csv_file):
    data_stream = from_csv(create_csv_file, variable="HeatFlux_st")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data,
        pd.DataFrame(
            {
                "HeatFlux_st": [
                    20.5,
                    21.0,
                    19.5,
                    22.0,
                    23.5,
                    24.0,
                    25.0,
                    26.5,
                    27.0,
                    28.0,
                ]
            }
        ),
    )


def test_from_csv_loads_time_column_when_requested(create_csv_file):
    data_stream = from_csv(create_csv_file, variable="time")

    assert isinstance(data_stream, DataStream)
    pd.testing.assert_frame_equal(
        data_stream.data,
        pd.DataFrame({"time": list(range(10))}),
    )


def test_from_csv_non_existent_file():
    non_existent_file = "non_existent_file.csv"

    with pytest.raises(
        ValueError, match=f"Error: file '{non_existent_file}' does not exist."
    ):
        from_csv(non_existent_file, variable="HeatFlux_st")


def test_from_csv_missing_variable_raises(create_csv_file):
    with pytest.raises(
        ValueError, match="Error: variable 'missing' does not exist in file"
    ):
        from_csv(create_csv_file, variable="missing")
