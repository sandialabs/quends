import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quends import DataStream

pytest_plugins = ("tests._shared",)


def test_init_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    assert len(ds) == 3
    assert ds.variables().tolist() == ["A"]


def test_init_empty(empty_data: pd.DataFrame):
    ds = DataStream(empty_data)
    assert len(ds) == 0
    assert ds.variables().tolist() == []


# ------------ mean --------------


def test_mean_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    assert ds.mean(window_size=1) == {"A": {"mean": 2.0, "window_size": 1}}


def test_mean_empty(empty_data: pd.DataFrame):
    ds = DataStream(empty_data)
    assert ds.mean() == {}


def test_mean_no_data(nan_data: pd.DataFrame):
    ds = DataStream(nan_data)
    result = ds.mean()

    for col in result:
        assert (
            "error" in result[col]
        ), f"Expected error for column '{col}', got {result[col]}"


def test_mean_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    assert ds.mean() == {
        "A": {"mean": 3.0, "window_size": 5},
        "B": {"mean": 3.0, "window_size": 5},
    }


def test_mean_long_overlapping_window(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    assert ds.mean() == {
        "A": {"mean": 3.0, "window_size": 5},
        "B": {"mean": 3.0, "window_size": 5},
    }


def test_mean_long_non_overlapping_window(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    assert ds.mean(method="non-overlapping", window_size=2) == {
        "A": {"mean": 2.5, "window_size": 2},
        "B": {"mean": 3.5, "window_size": 2},
    }


def test_estimate_window_falls_back_when_ess_is_invalid(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    column_data = ds.data["A"].dropna()
    original_effective_sample_size = ds.effective_sample_size

    ds.effective_sample_size = lambda *a, **k: {"results": {"A": object()}}

    result = ds._estimate_window("A", column_data, window_size=None)

    ds.effective_sample_size = original_effective_sample_size

    assert result == 5


# ------------ mean uncertainty --------------


def test_mean_uncertainty_no_data(nan_data: pd.DataFrame):
    ds = DataStream(nan_data)
    result = ds.mean_uncertainty()

    for col in result:
        assert "error" in result[col], f"No data available for column '{col}"


def test_mean_uncertainty_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    assert np.isnan(mean_uncertainty["A"]["mean_uncertainty"])
    assert mean_uncertainty["A"]["window_size"] == 2


def test_mean_uncertainty_sliding(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    result = ds.mean_uncertainty(method="sliding")

    assert "A" in result
    assert "mean_uncertainty" in result["A"]
    assert "window_size" in result["A"]
    assert result["A"]["mean_uncertainty"] >= 0


def test_mean_uncertainty_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    mean_uncertainty = ds.mean_uncertainty(window_size=2)
    assert mean_uncertainty == {
        "A": {"mean_uncertainty": 1.0, "window_size": 2},
        "B": {"mean_uncertainty": 1.0, "window_size": 2},
    }


# ------------ confidence interval --------------


def test_confidence_interval_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    expected = {
        "A": {
            "confidence_interval": (0.8683934723883333, 3.131606527611667),
            "window_size": 1,
        }
    }
    assert ds.confidence_interval(window_size=1) == expected


def test_confidence_interval_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    expected = {
        "A": {"confidence_interval": (0.54, 4.46), "window_size": 2},
        "B": {"confidence_interval": (1.54, 5.46), "window_size": 2},
    }
    assert ds.confidence_interval(window_size=2) == expected


def test_confidence_interval_no_data(nan_data: pd.DataFrame):
    ds = DataStream(nan_data)
    with pytest.raises(KeyError):
        ds.confidence_interval()


def test_confidence_interval_missing_data_for_column(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    original_mean = ds.mean
    original_mean_uncertainty = ds.mean_uncertainty

    ds.mean = lambda *a, **k: {"A": {"mean": 3.0, "window_size": 2}}
    ds.mean_uncertainty = lambda *a, **k: {
        "A": {"mean_uncertainty": 1.0, "window_size": 2}
    }

    result = ds.confidence_interval(column_name=["A", "B"], window_size=2)

    ds.mean = original_mean
    ds.mean_uncertainty = original_mean_uncertainty

    assert result["A"] == {
        "confidence_interval": (1.04, 4.96),
        "window_size": 2,
    }
    assert result["B"] == {"error": "Missing data for column 'B'"}


# ------------ compute stats --------------


def test_compute_stats_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    expected = {
        "A": {
            "mean": 2.0,
            "mean_uncertainty": 0.5773502691896258,
            "confidence_interval": (0.8683934723883333, 3.131606527611667),
            "pm_std": (1.4226497308103743, 2.5773502691896257),
            "effective_sample_size": 3,
            "window_size": 1,
        }
    }
    assert ds.compute_statistics(column_name="A", window_size=1) == expected


def test_compute_stats_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    expected = {
        "A": {
            "mean": 3.0,
            "mean_uncertainty": 0.7071067811865476,
            "confidence_interval": (1.6140707088743669, 4.385929291125633),
            "pm_std": (2.2928932188134525, 3.7071067811865475),
            "effective_sample_size": 5,
            "window_size": 1,
        }
    }
    assert ds.compute_statistics(column_name="A", window_size=1) == expected


def test_compute_stats_ci_not_computed(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    original_ci_method = ds.confidence_interval
    ds.confidence_interval = lambda *a, **k: {
        "A": {"confidence_interval": None, "window_size": 1}
    }
    result = ds.compute_statistics(column_name="A", window_size=1)
    ds.confidence_interval = original_ci_method
    assert "A" in result
    assert result["A"]["confidence_interval"] is None


def test_compute_statistics_missing_column(partial_nan_data: pd.DataFrame):
    ds = DataStream(partial_nan_data)

    ds.mean = lambda *a, **k: {"A": {"mean": 3.0, "window_size": 2}}
    ds.mean_uncertainty = lambda *a, **k: {
        "A": {"mean_uncertainty": 1.0, "window_size": 2}
    }
    ds.confidence_interval = lambda *a, **k: {
        "A": {"confidence_interval": (1.04, 4.96), "window_size": 2}
    }
    ds.effective_sample_size = lambda *a, **k: {"results": {"A": 10}}

    result = ds.compute_statistics(column_name=["A", "B"])

    assert "mean" in result["A"]
    assert result["B"] == {"error": "No data available for column 'B'"}


# ------------ cumulative stats --------------


def test_cumulative_stats_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    result = ds.cumulative_statistics(window_size=1)
    expected = {
        "A": {
            "cumulative_mean": [1.0, 1.5, 2.0],
            "cumulative_uncertainty": [np.nan, 0.7071067811865476, 1.0],
            "standard_error": [np.nan, 0.5, 0.5773502691896258],
            "window_size": 1,
        }
    }
    for key in expected["A"]:
        if isinstance(expected["A"][key], list):
            np.testing.assert_equal(result["A"][key], expected["A"][key])


def test_cumulative_stats_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.cumulative_statistics(window_size=1)
    expected = {
        "A": {
            "cumulative_mean": [1.0, 1.5, 2.0, 2.5, 3.0],
            "cumulative_uncertainty": [
                np.nan,
                0.7071067811865476,
                1.0,
                1.2909944487358056,
                1.5811388300841898,
            ],
            "standard_error": [
                np.nan,
                0.5,
                0.5773502691896258,
                0.6454972243679028,
                0.7071067811865476,
            ],
            "window_size": 1,
        },
        "B": {
            "cumulative_mean": [5.0, 4.5, 4.0, 3.5, 3.0],
            "cumulative_uncertainty": [
                np.nan,
                0.7071067811865476,
                1.0,
                1.2909944487358056,
                1.5811388300841898,
            ],
            "standard_error": [
                np.nan,
                0.5,
                0.5773502691896258,
                0.6454972243679028,
                0.7071067811865476,
            ],
            "window_size": 1,
        },
    }
    for col in ["A", "B"]:
        for key in expected[col]:
            np.testing.assert_equal(result[col][key], expected[col][key])


def test_cumulative_stats_empty(nan_data: pd.DataFrame):
    ds = DataStream(nan_data)
    expected = {"A": {"error": "No data available for column 'A'"}}
    assert ds.cumulative_statistics(window_size=1) == expected


def assert_nested_approx(a, b, rel=1e-8):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_approx(a[key], b[key], rel=rel)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_approx(left, right, rel=rel)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=rel)
    else:
        assert a == b


# ------------ additional data --------------


def test_additional_data_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    expected = {
        "A": {
            "A_est": 0.3910010411753345,
            "p_est": 0.8547556456757277,
            "n_current": 3,
            "current_sem": 0.1528818142001956,
            "target_sem": 0.13759363278017603,
            "n_target": 3.393548707049326,
            "additional_samples": 1,
            "window_size": 1,
        }
    }
    assert_nested_approx(result, expected)


def test_additional_data_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    expected = {
        "A": {
            "A_est": 0.3803501348616604,
            "p_est": 0.8838111262612045,
            "n_current": 5,
            "current_sem": 0.09171198805673249,
            "target_sem": 0.08254078925105925,
            "n_target": 5.633041271578334,
            "additional_samples": 1,
            "window_size": 1,
        },
        "B": {
            "A_est": 0.3803501348616604,
            "p_est": 0.8838111262612045,
            "n_current": 5,
            "current_sem": 0.09171198805673249,
            "target_sem": 0.08254078925105925,
            "n_target": 5.633041271578334,
            "additional_samples": 1,
            "window_size": 1,
        },
    }
    assert_nested_approx(result, expected)


def mock_cumulative_statistics_missing(col_name, method, window_size):
    return {"A": {"cumulative_uncertainty": [0.5, 0.4, 0.3]}, "B": {}}


def test_additional_data_missing_cumulative(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    ds.cumulative_statistics = mock_cumulative_statistics_missing
    additional_data = ds.additional_data(column_name="B", reduction_factor=0.1)
    expected = {"B": {"error": "No cumulative SEM data for column 'B'"}}
    assert additional_data == expected


def test_additional_data_not_enough_valid_points(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    ds.cumulative_statistics = lambda *a, **k: {
        "A": {"cumulative_uncertainty": [float("nan")]}
    }
    result = ds.additional_data(column_name="A")
    assert result["A"] == {"error": "Not enough valid data points for fitting."}


def test_additional_data_non_overlapping(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    result = ds.additional_data(column_name="A", method="non-overlapping")
    assert "additional_samples" in result["A"]
    assert "window_size" in result["A"]


def test_effective_sample_size_below_column_names_none(
    stationary_noise_df: pd.DataFrame,
):
    ds = DataStream(stationary_noise_df)
    result = ds.effective_sample_size_below(column_names=None)
    assert "A" in result
    assert "time" not in result
    assert result["A"] == 0


def test_effective_sample_size_below_single_column_string_is_normalized(
    long_data: pd.DataFrame,
):
    ds = DataStream(long_data)

    result = ds.effective_sample_size_below(column_names="B")

    assert result == {"B": 0}


@pytest.mark.parametrize(
    ("column_names", "expected"),
    [
        ("A", {"A": 0}),
        (["A", "B"], {"A": 0, "B": 0}),
    ],
)
def test_effective_sample_size_below_explicit_inputs(
    long_data: pd.DataFrame,
    column_names,
    expected,
):
    ds = DataStream(long_data)

    result = ds.effective_sample_size_below(column_names=column_names)

    assert result == expected


# ------------ estimate window --------------


def test_estimate_window_with_results_key(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    column_data = stationary_noise_df["A"].dropna()
    result = ds._estimate_window("A", column_data, window_size=None)
    assert result >= 5


def test_estimate_window_flat_ess_dict(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    ds.effective_sample_size = lambda *a, **k: {"A": 10}
    column_data = stationary_noise_df["A"].dropna()
    result = ds._estimate_window("A", column_data, window_size=None)
    assert result >= 5


def test_estimate_window_provided(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    column_data = stationary_noise_df["A"].dropna()
    result = ds._estimate_window("A", column_data, window_size=20)
    assert result == 20


# ------------ effective sample size --------------


def test_effective_sample_size_below_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    assert ds.effective_sample_size_below(column_names="A") == {"A": 0}


def test_effective_sample_size_below_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    assert ds.effective_sample_size_below(column_names="A") == {"A": 0}


def test_effective_sample_size_below_invalid_column(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    assert ds.effective_sample_size_below(column_names="C") == {"C": 0}


def test_effective_sample_size_below_empty_column():
    ds = DataStream(
        pd.DataFrame(
            {
                "time": [0, 1, 2, 3, 4],
                "A": [None, None, None, None, None],
                "B": [5, 4, 3, 2, 1],
            }
        )
    )
    assert ds.effective_sample_size_below(column_names="A") == {"A": 0}


def test_head(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    expected = pd.DataFrame(
        {"time": [0, 1, 2, 3, 4], "A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}
    )
    pd.testing.assert_frame_equal(ds.head(5), expected)


def test_process_column_missing_method(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    with pytest.raises(ValueError):
        ds._process_column(column_data="A", estimated_window=1, method="invalid_method")


def test_effective_sample_size_empty(empty_data: pd.DataFrame):
    ds = DataStream(empty_data)
    assert ds.effective_sample_size() == {"results": {}}


def test_effective_sample_size_nan(nan_data: pd.DataFrame):
    ds = DataStream(nan_data)
    expected = {
        "results": {
            "A": {
                "effective_sample_size": None,
                "message": "No data available for computation.",
            }
        }
    }
    assert ds.effective_sample_size(column_names=["A"]) == expected


def test_effective_sample_size_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    assert ds.effective_sample_size(column_names=["A"]) == {"results": {"A": 3}}


def test_effective_sample_size_long_data(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    assert ds.effective_sample_size(column_names=["A", "B"]) == {
        "results": {"A": 5, "B": 5}
    }


def test_effective_sample_size_stationary(stationary_data: pd.DataFrame):
    ds = DataStream(stationary_data)
    assert ds.effective_sample_size(column_names=["A"]) == {"results": {"A": 5}}


def test_effective_sample_size_trim_data(trim_data: pd.DataFrame):
    ds = DataStream(trim_data)
    assert ds.effective_sample_size(column_names=["A"]) == {"results": {"A": 5}}


def test_effective_sample_size_missing_col(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    expected = {
        "results": {"C": {"message": "Column 'C' not found in the DataStream."}}
    }
    assert ds.effective_sample_size(column_names=["C"]) == expected


# ------------  robust effective sample size --------------


def test_robust_effective_sample_size_too_few_samples():
    result = DataStream.robust_effective_sample_size([1.0, 2.0, 3.0], min_samples=8)
    assert np.isnan(result)


def test_robust_effective_sample_size_constant_signal_returns_full_length():
    result = DataStream.robust_effective_sample_size([5.0] * 10)
    assert result == 10.0


def test_robust_effective_sample_size_returns_relative_output():
    x = np.linspace(0.0, 1.0, 12)
    ess, relative = DataStream.robust_effective_sample_size(
        x,
        rank_normalize=False,
        return_relative=True,
    )

    assert 1.0 <= ess <= len(x)
    assert relative == pytest.approx(ess / len(x))


def test_robust_effective_sample_size_rank_normalized_path():
    x = np.array([10.0, 1.0, 7.0, 3.0, 9.0, 2.0, 8.0, 4.0, 6.0, 5.0])

    result = DataStream.robust_effective_sample_size(
        x,
        rank_normalize=True,
        min_samples=3,
    )

    assert 1.0 <= result <= len(x)


def test_robust_effective_sample_size_zero_variance_fallback():
    x = np.linspace(0.0, 1.0, 10)

    with patch("quends.base.data_stream.np.var", return_value=0.0):
        result = DataStream.robust_effective_sample_size(
            x,
            rank_normalize=False,
            min_samples=3,
            return_relative=True,
        )

    assert result == (10.0, 1.0)


def test_robust_effective_sample_size_loop_path_returns_bounded_ess():
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    result = DataStream.robust_effective_sample_size(
        x,
        rank_normalize=False,
        min_samples=3,
    )

    assert 1.0 <= result <= len(x)


# ------------ ess robust --------------


def test_ess_robust_returns_results_for_multiple_columns(long_data: pd.DataFrame):
    ds = DataStream(long_data)

    result = ds.ess_robust(column_names=["A", "B"], min_samples=3)

    assert set(result["results"]) == {"A", "B"}
    assert 1.0 <= result["results"]["A"] <= 5.0
    assert 1.0 <= result["results"]["B"] <= 5.0


def test_ess_robust_missing_column_reports_error(long_data: pd.DataFrame):
    ds = DataStream(long_data)

    result = ds.ess_robust(column_names=["A", "C"], min_samples=3)

    assert "A" in result["results"]
    assert result["results"]["C"] == {"error": "Column 'C' not found."}


def test_ess_robust_defaults_to_all_non_time_columns(long_data: pd.DataFrame):
    ds = DataStream(long_data)

    result = ds.ess_robust(column_names=None, min_samples=3)

    assert set(result["results"]) == {"A", "B"}


def test_ess_robust_relative_for_constant_signal(stationary_data: pd.DataFrame):
    ds = DataStream(stationary_data)

    result = ds.ess_robust(
        column_names="A",
        min_samples=3,
        return_relative=True,
    )

    assert result == {"results": {"A": (5.0, 1.0)}}
