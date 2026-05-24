import pandas as pd

from quends import DataStream, Ensemble


def test_compute_average_ensemble_ignores_empty_members():
    populated = DataStream(pd.DataFrame({"time": [0.0, 1.0], "A": [1.0, 3.0]}))
    empty = DataStream(pd.DataFrame(columns=["time", "A"]))

    average = Ensemble([empty, populated]).compute_average_ensemble()

    pd.testing.assert_frame_equal(
        average.data.reset_index(drop=True), populated.data.reset_index(drop=True)
    )


def test_compute_average_ensemble_preserves_empty_schema():
    empty = DataStream(pd.DataFrame(columns=["time", "A"]))

    average = Ensemble([empty]).compute_average_ensemble()

    pd.testing.assert_frame_equal(average.data, empty.data)
    assert average.mean() == {"A": {"error": "No data available for column 'A'"}}
