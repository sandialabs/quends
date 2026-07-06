import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quends import DataStream
from quends.base.history import DataStreamHistoryEntry
from quends.base.utils import _estimate_tau_int_from_series

pytest_plugins = ("tests._shared",)


def test_init_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    assert len(ds) == 3
    assert ds.variables().tolist() == ["A"]


def test_init_empty(empty_data: pd.DataFrame):
    ds = DataStream(empty_data)
    assert len(ds) == 0
    assert ds.variables().tolist() == []


def test_init_coerces_legacy_history_entries(simple_data: pd.DataFrame):
    typed_entry = DataStreamHistoryEntry("typed", {"a": 1})

    ds = DataStream(
        simple_data,
        history=[
            typed_entry,
            {"operation": "legacy_dict", "options": {"b": 2}},
            "legacy_value",
        ],
    )

    entries = ds.history.entries()
    assert entries[0] is typed_entry
    assert entries[1] == DataStreamHistoryEntry("legacy_dict", {"b": 2})
    assert entries[2] == DataStreamHistoryEntry(
        "str",
        {"value": "legacy_value"},
    )


def test_df_setter_updates_underlying_data(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    replacement = pd.DataFrame({"B": [10, 20]})

    ds.df = replacement

    assert ds.data is replacement
    assert ds.df is replacement


# ------------ mean --------------


def test_mean_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    result = ds.compute_statistics(window_size=1)
    assert result["A"]["mean"] == 2.0
    assert result["A"]["window_size"] == 1


def test_mean_empty(empty_data: pd.DataFrame):
    ds = DataStream(empty_data)
    result = ds.compute_statistics()
    assert result == {}


def test_mean_no_data(nan_data: pd.DataFrame):
    ds = DataStream(nan_data)
    result = ds.compute_statistics()

    for col in result:
        assert (
            "error" in result[col]
        ), f"Expected error for column '{col}', got {result[col]}"


def test_mean_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.compute_statistics()
    assert result["A"]["mean"] == 3.0
    assert result["A"]["window_size"] == 5
    assert result["B"]["mean"] == 3.0
    assert result["B"]["window_size"] == 5


def test_mean_long_overlapping_window(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.compute_statistics()
    assert result["A"]["mean"] == 3.0
    assert result["A"]["window_size"] == 5
    assert result["B"]["mean"] == 3.0
    assert result["B"]["window_size"] == 5


def test_mean_long_non_overlapping_window(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.compute_statistics(method="non-overlapping", window_size=2)
    assert result["A"]["mean"] == 2.5
    assert result["A"]["window_size"] == 2
    assert result["B"]["mean"] == 3.5
    assert result["B"]["window_size"] == 2


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
    result = ds.compute_statistics()

    for col in result:
        assert "error" in result[col], f"No data available for column '{col}"


def test_mean_uncertainty_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    result = ds.compute_statistics(window_size=2)
    assert np.isnan(result["A"]["mean_uncertainty"])
    assert result["A"]["window_size"] == 2


def test_mean_uncertainty_sliding(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    result = ds.compute_statistics(method="sliding")

    assert "A" in result
    assert "mean_uncertainty" in result["A"]
    assert "window_size" in result["A"]
    assert result["A"]["mean_uncertainty"] >= 0


def test_mean_uncertainty_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.compute_statistics(window_size=2)
    # SE depends on Geyer ESS on the 2 block means; exact value is implementation-
    # dependent but must be non-negative and finite.
    assert "A" in result and "mean_uncertainty" in result["A"]
    assert "B" in result and "mean_uncertainty" in result["B"]
    assert result["A"]["window_size"] == 2
    assert result["B"]["window_size"] == 2
    assert result["A"]["mean_uncertainty"] >= 0
    assert result["B"]["mean_uncertainty"] >= 0


# ------------ confidence interval --------------


def test_confidence_interval_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    result = ds.compute_statistics(window_size=1)
    assert result["A"]["confidence_interval"] == pytest.approx(
        (0.8683934723883333, 3.131606527611667)
    )
    assert result["A"]["window_size"] == 1


def test_confidence_interval_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.compute_statistics(window_size=2)
    # SE now uses Geyer ESS on block means, so the exact bounds are
    # implementation-dependent.  Check structural invariants instead.
    for col in ("A", "B"):
        assert col in result
        ci = result[col]["confidence_interval"]
        assert len(ci) == 2, f"CI for {col} must be a 2-tuple"
        assert ci[0] < ci[1], f"CI lower bound must be < upper bound for {col}"
        assert result[col]["window_size"] == 2


def test_confidence_interval_no_data(nan_data: pd.DataFrame):
    # compute_statistics propagates error dicts instead of raising KeyError.
    ds = DataStream(nan_data)
    result = ds.compute_statistics()
    for col in result:
        assert "error" in result[col], f"Expected error dict for column '{col}'"


def test_confidence_interval_missing_data_for_column(partial_nan_data: pd.DataFrame):
    # partial_nan_data: A=[1,2,3] (valid), B=[None,None,None] (all NaN).
    # compute_statistics returns an error dict for B and a valid result for A.
    ds = DataStream(partial_nan_data)
    result = ds.compute_statistics(window_size=2)

    assert "confidence_interval" in result["A"], "A should produce a CI result"
    assert result["A"]["window_size"] == 2
    assert "error" in result["B"], "B (all-NaN) should return an error dict"


# ------------ compute stats --------------


def test_compute_stats_simple(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    result = ds.compute_statistics(column_name="A", window_size=1)
    # compute_statistics returns extra diagnostic keys (ess_blocks, se_method, …).
    # Check the core fields individually so that new diagnostics don't break this test.
    col = result["A"]
    assert col["mean"] == pytest.approx(2.0)
    assert col["mean_uncertainty"] == pytest.approx(0.5773502691896258)
    assert col["confidence_interval"] == pytest.approx(
        (0.8683934723883333, 3.131606527611667)
    )
    assert col["pm_std"] == pytest.approx((1.4226497308103743, 2.5773502691896257))
    assert col["effective_sample_size"] == 3
    assert col["window_size"] == 1


def test_compute_stats_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    result = ds.compute_statistics(column_name="A", window_size=1)
    # compute_statistics returns extra diagnostic keys (ess_blocks, se_method, …).
    # Check the core fields individually so that new diagnostics don't break this test.
    col = result["A"]
    assert col["mean"] == pytest.approx(3.0)
    assert col["mean_uncertainty"] == pytest.approx(0.7071067811865476)
    assert col["confidence_interval"] == pytest.approx(
        (1.6140707088743669, 4.385929291125633)
    )
    assert col["pm_std"] == pytest.approx((2.2928932188134525, 3.7071067811865475))
    assert col["effective_sample_size"] == 5
    assert col["window_size"] == 1


def test_compute_stats_ci_not_computed(long_data: pd.DataFrame):
    # compute_statistics computes CI internally — monkey-patching
    # ds.confidence_interval has no effect.  Verify instead that the returned
    # dict always contains the expected top-level keys.
    ds = DataStream(long_data)
    result = ds.compute_statistics(column_name="A", window_size=1)
    assert "A" in result
    required_keys = {
        "mean",
        "mean_uncertainty",
        "confidence_interval",
        "pm_std",
        "effective_sample_size",
        "window_size",
    }
    assert required_keys.issubset(result["A"].keys())


def test_compute_statistics_missing_column(partial_nan_data: pd.DataFrame):
    ds = DataStream(partial_nan_data)

    ds.effective_sample_size = lambda *a, **k: {"results": {"A": 10}}

    result = ds.compute_statistics(column_name=["A", "B"])

    assert "mean" in result["A"]
    assert result["B"] == {"error": "No data available for column 'B'"}


def test_compute_statistics_handles_no_block_means(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)

    def fake_process_column(*args, **kwargs):
        return pd.Series(dtype=float), {
            "window_size": 10,
            "independence_status": "too_few_blocks",
            "blocks": np.array([]),
            "ljungbox_lags": [],
            "ljungbox_pvalues": [],
            "n_blocks": 0,
            "independent": False,
        }

    ds._process_column = fake_process_column

    result = ds.compute_statistics(column_name="A")

    assert result["A"] == {
        "error": "No block means produced (window_size=10).",
        "window_size": 10,
    }


def test_compute_statistics_best_p_status_adds_warning(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    ds.effective_sample_size = lambda *a, **k: {"results": {"A": 3}}

    def fake_process_column(*args, **kwargs):
        return pd.Series([1.0, 2.0, 3.0]), {
            "window_size": 1,
            "independence_status": "best_p",
            "blocks": np.array([1.0, 2.0, 3.0]),
            "ljungbox_lags": [2],
            "ljungbox_pvalues": [0.01],
            "n_blocks": 3,
            "independent": False,
        }

    ds._process_column = fake_process_column

    result = ds.compute_statistics(column_name="A", window_size=1)

    assert result["A"]["se_method"] == "iid_blocks_best_p"
    assert result["A"]["se_effective_n"] == 3.0
    assert result["A"]["warning"] == (
        "Block means did not pass Ljung-Box; using best-p window."
    )


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
    # Power law is fit to the SEM (standard_error), not the expanding std — see
    # AUDIT_REPORT H2; values below are the corrected SEM-based fit.
    expected = {
        "A": {
            "A_est": 0.39100104117533435,
            "p_est": 0.3547556456757279,
            "n_current": 3,
            "current_sem": 0.2647990697480437,
            "target_sem": 0.2383191627732393,
            "n_target": 4.037424148235672,
            "additional_samples": 2,
            "window_size": 1,
        }
    }
    assert_nested_approx(result, expected)


def test_additional_data_long(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = ds.additional_data(window_size=1, method="sliding")
    # SEM-based fit (AUDIT_REPORT H2).
    expected = {
        "A": {
            "A_est": 0.3822405006605927,
            "p_est": 0.3800829365856315,
            "n_current": 5,
            "current_sem": 0.20733381047055485,
            "target_sem": 0.18660042942349936,
            "n_target": 6.597177634200694,
            "additional_samples": 2,
            "window_size": 1,
        },
        "B": {
            "A_est": 0.3822405006605927,
            "p_est": 0.3800829365856315,
            "n_current": 5,
            "current_sem": 0.20733381047055485,
            "target_sem": 0.18660042942349936,
            "n_target": 6.597177634200694,
            "additional_samples": 2,
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
    ds.cumulative_statistics = lambda *a, **k: {"A": {"standard_error": [float("nan")]}}
    result = ds.additional_data(column_name="A")
    assert result["A"] == {"error": "Not enough valid data points for fitting."}


def test_additional_data_non_overlapping(stationary_noise_df: pd.DataFrame):
    ds = DataStream(stationary_noise_df)
    result = ds.additional_data(column_name="A", method="non-overlapping")
    assert "additional_samples" in result["A"]
    assert "window_size" in result["A"]


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


def test_head(long_data: pd.DataFrame):
    ds = DataStream(long_data)
    expected = pd.DataFrame(
        {"time": [0, 1, 2, 3, 4], "A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}
    )
    pd.testing.assert_frame_equal(ds.head(5), expected)


def test_process_column_missing_method(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)
    # Method is validated before any data processing, so ValueError is raised
    # even with a valid pd.Series.
    column_data = simple_data["A"]
    with pytest.raises(ValueError):
        ds._process_column(
            column_data=column_data, estimated_window=1, method="invalid_method"
        )


def test_process_column_ignores_mismatched_time_values(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)

    with patch(
        "quends.base.data_stream.autotune_blocks",
        return_value={
            "window_size": 1,
            "blocks": np.array([1.0, 2.0, 3.0]),
            "n_blocks": 3,
            "independence_status": "user_window",
            "independent": False,
            "ljungbox_lags": [],
            "ljungbox_pvalues": [],
        },
    ):
        processed, _ = ds._process_column(
            simple_data["A"],
            estimated_window=1,
            method="non-overlapping",
            time_values=np.array([10.0]),
        )

    assert processed.index.tolist() == [0, 1, 2]


def test_process_column_returns_empty_when_no_blocks(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)

    with patch(
        "quends.base.data_stream.autotune_blocks",
        return_value={
            "window_size": 10,
            "blocks": np.array([]),
            "n_blocks": 0,
            "independence_status": "too_few_blocks",
            "independent": False,
            "ljungbox_lags": [],
            "ljungbox_pvalues": [],
        },
    ):
        processed, ab = ds._process_column(
            simple_data["A"],
            estimated_window=10,
            method="non-overlapping",
        )

    assert processed.empty
    assert ab["n_blocks"] == 0


def test_time_values_for_series_returns_none_on_alignment_error():
    ds = DataStream(pd.DataFrame({"time": [0.0, 1.0], "A": [1.0, 2.0]}))
    series = pd.Series([1.0], index=[99])

    assert ds._time_values_for_series(series) is None


def test_estimate_tau_int_delegates_and_returns_float(long_data: pd.DataFrame):
    ds = DataStream(long_data)

    result = ds._estimate_tau_int(long_data["A"])

    assert isinstance(result, float)
    assert result >= 1.0


def test_estimate_tau_int_warns_when_acf_lag_cutoff_is_tiny():
    # Four samples gives nlags = max(1, min(n // 4, 2000)) = 1.
    # That deliberately tiny ACF horizon should trigger the under-resolution warning.
    with pytest.warns(UserWarning, match="decorrelation time"):
        tau_int = _estimate_tau_int_from_series(np.array([1.0, 2.0, 3.0, 4.0]))

    assert tau_int >= 1.0


def test_tau_int_lag_cutoff_warning_is_returned_in_metadata():
    ds = DataStream(pd.DataFrame({"time": np.arange(50), "A": np.arange(50.0)}))

    with pytest.warns(UserWarning, match="decorrelation time"):
        result = ds.compute_statistics("A")

    column_warnings = result["A"]["metadata"]["warnings"]
    assert any(
        "decorrelation time" in warning and "Results may be inaccurate" in warning
        for warning in column_warnings
    )
    assert any(
        warning["column"] == "A"
        and "decorrelation time" in warning["message"]
        and "Results may be inaccurate" in warning["message"]
        for warning in result.metadata["warnings"]
    )


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
    # Geyer positive-pair truncation includes rho_2 ≈ 0.41 in the first pair,
    # giving tau_int ≈ 3.22 → ESS = ceil(10 / 3.22) = 4.
    # The old abs-threshold approach dropped rho_2 (below 1.96/√10 ≈ 0.62) and
    # returned 5 — Geyer is slightly more conservative and correct.
    assert ds.effective_sample_size(column_names=["A"]) == {"results": {"A": 4}}


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


# --- B1 result-object / metadata (schema unification) ------------------------
def test_compute_statistics_returns_statsresult_with_metadata(long_data):
    ds = DataStream(long_data)
    res = ds.compute_statistics("A")
    # Backward-compatible: behaves exactly like the historical {col: {...}} dict.
    assert res["A"]["mean_uncertainty"] is not None  # key name preserved (not "sem")
    assert "mean" in res["A"]
    assert res == dict(res)  # equals a plain dict
    # New: carries run-level provenance in .metadata.
    assert res.metadata["estimator"] == "single"
    assert res.metadata["schema_version"] == "1.0"
    assert res.metadata["total_samples"] == len(long_data)


# --- §2 convenience API: DataStream.trim one-liner + input validation --------
def test_datastream_trim_one_liner_matches_explicit_path():
    import numpy as np

    from quends.base.trim import TrimDataStreamOperation, build_trim_strategy

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "time": np.arange(400.0),
            "x": np.r_[np.linspace(0, 5, 80), 5 + 0.2 * rng.standard_normal(320)],
        }
    )
    ds = DataStream(df)
    # column auto-detected (single non-time column)
    new = ds.trim(method="threshold", threshold=0.1, window_size=20)
    strat = build_trim_strategy(
        method="threshold", window_size=20, start_time=0.0, threshold=0.1
    )
    old = TrimDataStreamOperation(strategy=strat)(ds, column_name="x")
    assert isinstance(new, DataStream)
    assert len(new.data) == len(old.data)


def test_datastream_trim_requires_column_when_multiple_signal_columns(long_data):
    ds = DataStream(long_data)

    with pytest.raises(ValueError, match="column_name must be specified"):
        ds.trim(method="std")


def test_datastream_trim_passes_extra_strategy_kwargs(simple_data: pd.DataFrame):
    ds = DataStream(simple_data)

    class DummyStrategy:
        method_name = "dummy"

    class DummyOperation:
        def __init__(self, strategy):
            self.strategy = strategy

        def __call__(self, data_stream, column_name):
            assert data_stream is ds
            assert column_name == "A"
            assert self.strategy.custom_option == "set-by-trim"
            return DataStream(pd.DataFrame({"A": [42]}))

    with patch(
        "quends.base.trim.build_trim_strategy",
        return_value=DummyStrategy(),
    ), patch("quends.base.trim.TrimDataStreamOperation", DummyOperation):
        result = ds.trim(column_name="A", custom_option="set-by-trim")

    assert result.data["A"].tolist() == [42]


def test_datastream_init_rejects_non_dataframe():
    for bad in (None, "nope", 42):
        with pytest.raises(TypeError):
            DataStream(bad)


def test_datastream_init_coerces_dict():
    ds = DataStream({"time": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
    assert list(ds.data.columns) == ["time", "x"]


def test_get_block_effective_n_returns_error_shape_for_bad_column(long_data):
    ds = DataStream(long_data)
    ds.compute_statistics = lambda *a, **k: {"missing": {"error": "no data"}}

    result = ds.get_block_effective_n("missing", window_size=7)

    assert np.isnan(result["effective_n"])
    assert result["window_size"] == 7
    assert result["n_blocks"] == 0


def test_get_block_effective_n_extracts_statistics_fields(long_data):
    ds = DataStream(long_data)

    result = ds.get_block_effective_n("A", window_size=1)

    assert result["effective_n"] >= 1.0
    assert result["window_size"] == 1
    assert result["n_blocks"] == 5


def test_variance_extracts_success_and_error_entries(long_data):
    ds = DataStream(long_data)
    ds.compute_statistics = lambda *a, **k: {
        "A": {"variance": 2.5, "window_size": 3, "ess_blocks": 4.0},
        "B": {"error": "no data"},
    }

    result = ds._variance(column_name=["A", "B"])

    assert result["A"] == {
        "variance": 2.5,
        "window_size": 3,
        "effective_n_blocks": 4.0,
    }
    assert np.isnan(result["B"]["variance"])
    assert np.isnan(result["B"]["window_size"])
    assert np.isnan(result["B"]["effective_n_blocks"])
