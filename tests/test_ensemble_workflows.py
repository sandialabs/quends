"""
test_ensemble_workflows.py
--------------------------
Smoke-tests for:

  - EnsembleAverageWorkflow
  - EnsembleStatisticsWorkflow
  - BatchEnsembleWorkflow (scaffold — NotImplementedError expected)

Also tests the reusable helper modules:
  - quends.base.ensemble_utils
  - quends.base.ensemble_statistics

Synthetic data
--------------
Three or four members with:
  a) the same time grid (no interpolation needed)
  b) slightly different time grids (interpolation path)
  c) a short transient followed by a stationary segment
"""

import numpy as np
import pandas as pd
import pytest

from quends import (
    DataStream,
    Ensemble,
    BatchEnsembleWorkflow,
    EnsembleAverageWorkflow,
    EnsembleStatisticsWorkflow,
)
from quends.base.ensemble_utils import (
    check_time_steps_uniformity,
    compute_average_ensemble,
    direct_average,
    get_common_variables,
    interpolate_to_common_time,
    resolve_cols,
    trim_members,
    validate_column,
    validate_members,
)
from quends.base.ensemble_statistics import (
    compute_ensemble_statistics,
    ensemble_average_stats_for_col,
    ivw_member_means_stats_for_col,
    pool_block_means,
    pooled_block_means_stats_for_col,
)
from quends.workflow import (
    BatchEnsembleWorkflow,
    EnsembleAverageWorkflow,
    EnsembleStatisticsWorkflow,
    RobustWorkflow,
)


# ---------------------------------------------------------------------------
# Shared synthetic-data factories
# ---------------------------------------------------------------------------

def _make_stationary_member(seed: int, n: int = 300, mean: float = 2.0) -> DataStream:
    """Return a DataStream with a uniform time grid and stationary signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 30.0, n)
    y = mean + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"time": t, "signal": y, "noise": rng.standard_normal(n)})
    return DataStream(df)


def _make_transient_member(seed: int, n: int = 400, mean: float = 3.0) -> DataStream:
    """Return a DataStream with a transient followed by a stationary tail."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 40.0, n)
    transient = 5.0 * np.exp(-t / 5.0)
    steady = mean + rng.standard_normal(n) * 0.4
    y = transient + steady
    df = pd.DataFrame({"time": t, "signal": y})
    return DataStream(df)


def _make_different_grid_member(seed: int, n: int = 280, mean: float = 2.0) -> DataStream:
    """Return a DataStream with a slightly different time grid."""
    rng = np.random.default_rng(seed)
    # Start slightly later, end slightly later
    t = np.linspace(0.5, 30.5, n)
    y = mean + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"time": t, "signal": y})
    return DataStream(df)


# Convenience fixtures
@pytest.fixture
def same_grid_members():
    """Three members sharing the same uniform time grid."""
    return [_make_stationary_member(i, n=300) for i in range(3)]


@pytest.fixture
def transient_members():
    """Three members with transients followed by stationary tails."""
    return [_make_transient_member(i, n=400) for i in range(3)]


@pytest.fixture
def different_grid_members():
    """Mix of members with different time grids."""
    m0 = _make_stationary_member(0, n=300)
    m1 = _make_different_grid_member(1, n=280)
    m2 = _make_stationary_member(2, n=300)
    return [m0, m1, m2]


# ---------------------------------------------------------------------------
# Tests: ensemble_utils helpers
# ---------------------------------------------------------------------------

class TestValidateMembers:
    def test_valid(self, same_grid_members):
        validate_members(same_grid_members)  # should not raise

    def test_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_members([])

    def test_not_a_list(self):
        with pytest.raises(TypeError):
            validate_members("not_a_list")

    def test_non_datastream_member(self, same_grid_members):
        bad = same_grid_members + ["oops"]
        with pytest.raises(TypeError, match="DataStream"):
            validate_members(bad)


class TestValidateColumn:
    def test_valid(self, same_grid_members):
        validate_column(same_grid_members, "signal")  # should not raise

    def test_missing_column(self, same_grid_members):
        with pytest.raises(KeyError, match="nonexistent"):
            validate_column(same_grid_members, "nonexistent")

    def test_non_string_column(self, same_grid_members):
        with pytest.raises(TypeError):
            validate_column(same_grid_members, 42)


class TestGetCommonVariables:
    def test_returns_sorted_common(self, same_grid_members):
        cols = get_common_variables(same_grid_members)
        # "noise" and "signal" both present in all three members
        assert "signal" in cols
        assert "noise" in cols
        assert "time" not in cols

    def test_empty_list(self):
        assert get_common_variables([]) == []

    def test_disjoint_columns(self):
        m0 = DataStream(pd.DataFrame({"time": [0.0, 1.0], "A": [1.0, 2.0]}))
        m1 = DataStream(pd.DataFrame({"time": [0.0, 1.0], "B": [3.0, 4.0]}))
        assert get_common_variables([m0, m1]) == []


class TestResolveCols:
    def test_string(self, same_grid_members):
        assert resolve_cols(same_grid_members, "signal") == ["signal"]

    def test_none(self, same_grid_members):
        cols = resolve_cols(same_grid_members, None)
        assert "signal" in cols

    def test_list(self, same_grid_members):
        assert resolve_cols(same_grid_members, ["signal", "noise"]) == ["signal", "noise"]


class TestCheckTimeStepsUniformity:
    def test_uniform(self, same_grid_members):
        info = check_time_steps_uniformity(same_grid_members)
        assert info["uniform"] is True
        assert np.isfinite(info["majority_step"])

    def test_non_uniform(self, different_grid_members):
        info = check_time_steps_uniformity(different_grid_members)
        # At least one of majority_step should be finite
        assert np.isfinite(info["majority_step"])

    def test_member_keys(self, same_grid_members):
        info = check_time_steps_uniformity(same_grid_members)
        assert "Member 0" in info["members"]
        assert "status" in info["members"]["Member 0"]


class TestInterpolateToCommonTime:
    def test_same_grid_runs(self, same_grid_members):
        new_members, diag = interpolate_to_common_time(same_grid_members)
        assert len(new_members) == len(same_grid_members)
        assert isinstance(new_members[0], DataStream)
        assert "n_grid" in diag

    def test_different_grids(self, different_grid_members):
        new_members, diag = interpolate_to_common_time(different_grid_members)
        assert len(new_members) == len(different_grid_members)
        # All on same grid now
        t_new = [m.data["time"].values for m in new_members]
        for t in t_new[1:]:
            np.testing.assert_allclose(t, t_new[0], atol=1e-6)

    def test_unsupported_method_falls_back_to_linear(self, different_grid_members):
        # Passing "linear" should work
        new_members, _ = interpolate_to_common_time(different_grid_members, method="linear")
        assert len(new_members) == len(different_grid_members)


class TestDirectAverage:
    def test_returns_datastream(self, same_grid_members):
        avg_ds, meta = direct_average(same_grid_members)
        assert isinstance(avg_ds, DataStream)
        assert not avg_ds.data.empty
        assert "signal" in avg_ds.data.columns

    def test_metadata(self, same_grid_members):
        _, meta = direct_average(same_grid_members)
        assert meta["n_members"] == len(same_grid_members)

    def test_mean_between_extremes(self, same_grid_members):
        avg_ds, _ = direct_average(same_grid_members, cols=["signal"])
        avg_val = avg_ds.data["signal"].mean()
        assert 1.0 < avg_val < 3.0  # planted mean=2.0 with noise


class TestComputeAverageEnsemble:
    def test_same_grid(self, same_grid_members):
        avg = compute_average_ensemble(same_grid_members)
        assert isinstance(avg, DataStream)
        assert not avg.data.empty

    def test_different_grids(self, different_grid_members):
        avg = compute_average_ensemble(different_grid_members, verbose=False)
        assert isinstance(avg, DataStream)
        assert not avg.data.empty

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No data streams"):
            compute_average_ensemble([])


class TestTrimMembers:
    def test_returns_list(self, transient_members):
        kept = trim_members(transient_members, column_name="signal", method="std",
                            window_size=50, start_time=0.0)
        assert isinstance(kept, list)
        assert all(isinstance(m, DataStream) for m in kept)

    def test_all_survived(self, transient_members):
        kept = trim_members(transient_members, column_name="signal", method="std",
                            window_size=50, start_time=0.0)
        # At least some members survive a transient trim
        assert len(kept) >= 1

    def test_invalid_method(self, transient_members):
        with pytest.raises(ValueError):
            trim_members(transient_members, column_name="signal", method="invalid_method")


# ---------------------------------------------------------------------------
# Tests: ensemble_statistics helpers
# ---------------------------------------------------------------------------

class TestPoolBlockMeans:
    def test_returns_array_and_meta(self, transient_members):
        from quends.base.trim import build_trim_strategy, TrimDataStreamOperation
        strategy = build_trim_strategy("std", window_size=50, start_time=0.0)
        op = TrimDataStreamOperation(strategy=strategy)
        trimmed = [op(m, column_name="signal") for m in transient_members]
        trimmed = [t for t in trimmed if not t.data.empty]
        if trimmed:
            pooled, meta = pool_block_means(trimmed, "signal")
            assert isinstance(pooled, np.ndarray)
            assert "members_used" in meta


class TestEnsembleAverageStatsForCol:
    def test_runs_on_stationary(self, same_grid_members):
        stat, meta = ensemble_average_stats_for_col(same_grid_members, "signal")
        # Even if the column isn't present in avg, we get a dict back
        assert isinstance(stat, dict)
        assert isinstance(meta, dict)

    def test_mean_is_finite(self, same_grid_members):
        stat, _ = ensemble_average_stats_for_col(same_grid_members, "signal")
        assert np.isfinite(stat.get("mean", np.nan))


class TestPooledBlockMeansStatsForCol:
    def test_runs(self, same_grid_members):
        stat, meta = pooled_block_means_stats_for_col(same_grid_members, "signal")
        assert isinstance(stat, dict)
        assert "mean" in stat
        assert "confidence_interval" in stat

    def test_mean_finite(self, same_grid_members):
        stat, _ = pooled_block_means_stats_for_col(same_grid_members, "signal")
        assert np.isfinite(stat["mean"])


class TestIvwMemberMeansStatsForCol:
    def test_runs(self, same_grid_members):
        stat, meta = ivw_member_means_stats_for_col(same_grid_members, "signal")
        assert isinstance(stat, dict)
        assert "mean" in stat
        assert "se_method" in stat

    def test_se_method(self, same_grid_members):
        stat, _ = ivw_member_means_stats_for_col(same_grid_members, "signal")
        assert stat["se_method"] == "ivw_member_means"

    def test_full_diagnostics(self, same_grid_members):
        stat, _ = ivw_member_means_stats_for_col(
            same_grid_members, "signal", diagnostics="full"
        )
        assert stat.get("individual") is not None
        assert "Member 0" in stat["individual"]


class TestComputeEnsembleStatistics:
    def test_ensemble_average(self, same_grid_members):
        out = compute_ensemble_statistics(
            same_grid_members, "signal", technique="ensemble_average"
        )
        assert "results" in out
        assert "signal" in out["results"]
        assert "technique_0_ensemble_average" in out["metadata"]

    def test_pooled_block_means(self, same_grid_members):
        out = compute_ensemble_statistics(
            same_grid_members, "signal", technique="pooled_block_means"
        )
        assert "results" in out
        assert "signal" in out["results"]
        assert "technique_1_pooled_block_means" in out["metadata"]

    def test_ivw_member_means(self, same_grid_members):
        out = compute_ensemble_statistics(
            same_grid_members, "signal", technique="ivw_member_means"
        )
        assert "results" in out
        assert "signal" in out["results"]
        assert "technique_2_ivw_member_means" in out["metadata"]

    # ── Backward-compat: legacy integer & 'techniqueN' aliases ─────────────
    def test_legacy_integer_aliases(self, same_grid_members):
        out0 = compute_ensemble_statistics(same_grid_members, "signal", technique=0)
        out1 = compute_ensemble_statistics(same_grid_members, "signal", technique=1)
        out2 = compute_ensemble_statistics(same_grid_members, "signal", technique=2)
        assert "technique_0_ensemble_average" in out0["metadata"]
        assert "technique_1_pooled_block_means" in out1["metadata"]
        assert "technique_2_ivw_member_means" in out2["metadata"]

    def test_legacy_string_aliases(self, same_grid_members):
        out1 = compute_ensemble_statistics(
            same_grid_members, "signal", technique="technique1"
        )
        out2 = compute_ensemble_statistics(
            same_grid_members, "signal", technique="technique2"
        )
        assert "technique_1_pooled_block_means" in out1["metadata"]
        assert "technique_2_ivw_member_means" in out2["metadata"]

    def test_invalid_technique(self, same_grid_members):
        with pytest.raises(ValueError, match="Invalid technique"):
            compute_ensemble_statistics(same_grid_members, "signal", technique=99)

    def test_none_column_resolves(self, same_grid_members):
        out = compute_ensemble_statistics(
            same_grid_members, None, technique="pooled_block_means"
        )
        # All common variables should be present
        cols = set(out["results"].keys())
        assert "signal" in cols


# ---------------------------------------------------------------------------
# Tests: Ensemble class backward compatibility
# ---------------------------------------------------------------------------

class TestEnsembleBackwardCompat:
    def test_compute_statistics_ensemble_average(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        out = ens.compute_statistics(
            column_name="signal", technique="ensemble_average"
        )
        assert "results" in out
        assert np.isfinite(out["results"]["signal"].get("mean", np.nan))

    def test_compute_statistics_pooled_block_means(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        out = ens.compute_statistics(
            column_name="signal", technique="pooled_block_means"
        )
        assert "results" in out

    def test_compute_statistics_ivw_member_means(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        out = ens.compute_statistics(
            column_name="signal", technique="ivw_member_means"
        )
        assert "results" in out
        assert out["results"]["signal"]["se_method"] == "ivw_member_means"

    # Backward-compat: legacy integer aliases must still resolve correctly.
    def test_compute_statistics_legacy_integer(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        for legacy, expected_se in (
            (1, None),  # pooled_block_means: no fixed se_method assertion
            (2, "ivw_member_means"),
        ):
            out = ens.compute_statistics(column_name="signal", technique=legacy)
            assert "results" in out
            if expected_se is not None:
                assert out["results"]["signal"]["se_method"] == expected_se

    def test_compute_average_ensemble(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        avg = ens.compute_average_ensemble()
        assert isinstance(avg, DataStream)
        assert not avg.data.empty

    def test_check_time_steps_uniformity(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        info = ens.check_time_steps_uniformity()
        assert info["uniform"] is True

    def test_interpolate_to_common_time(self, different_grid_members):
        ens = Ensemble(different_grid_members)
        new_ens, diag = ens.interpolate_to_common_time()
        assert isinstance(new_ens, Ensemble)
        assert "n_grid" in diag

    def test_common_variables(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        cols = ens.common_variables()
        assert "signal" in cols
        assert "time" not in cols

    def test_trim_returns_ensemble(self, transient_members):
        ens = Ensemble(transient_members)
        trimmed_ens = ens.trim("signal", method="std", window_size=50)
        assert isinstance(trimmed_ens, Ensemble)
        assert len(trimmed_ens) >= 1

    def test_validate_members_called(self):
        with pytest.raises((ValueError, TypeError)):
            Ensemble([])

    def test_invalid_member_type(self):
        with pytest.raises((ValueError, TypeError)):
            Ensemble(["not_a_datastream"])


# ---------------------------------------------------------------------------
# Tests: EnsembleAverageWorkflow
# ---------------------------------------------------------------------------

class TestEnsembleAverageWorkflow:

    def test_import(self):
        from quends.workflow import EnsembleAverageWorkflow  # noqa: F401

    def test_top_level_import(self):
        from quends import EnsembleAverageWorkflow  # noqa: F401

    def test_run_same_grid(self, same_grid_members):
        wf = EnsembleAverageWorkflow(
            column_name="signal",
            trim_method="std",
            window_size=50,
            start_time=0.0,
        )
        result = wf.run(same_grid_members)

        assert result["workflow"] == "ensemble_average"
        assert result["n_members"] == len(same_grid_members)
        assert result["interpolation_required"] is False
        assert result["common_grid"] is None
        assert isinstance(result["trimmed_stream"], DataStream)
        assert not result["trimmed_stream"].data.empty
        assert isinstance(result["statistics"], dict)
        assert "signal" in result["statistics"]
        assert np.isfinite(result["statistics"]["signal"].get("mean", np.nan))

    def test_run_different_grids_triggers_interpolation(self):
        # Members with DIFFERENT step sizes → interpolation_required=True.
        # Use low noise so the averaged trace passes ADF stationarity test.
        rng = np.random.default_rng(42)
        m0 = DataStream(pd.DataFrame({
            "time": np.linspace(0.0, 100.0, 1000),   # dt ≈ 0.1001
            "signal": 2.0 + rng.standard_normal(1000) * 0.05,
        }))
        m1 = DataStream(pd.DataFrame({
            "time": np.linspace(0.0, 100.0, 900),    # dt ≈ 0.1112 — different!
            "signal": 2.0 + rng.standard_normal(900) * 0.05,
        }))
        m2 = DataStream(pd.DataFrame({
            "time": np.linspace(0.0, 100.0, 1000),
            "signal": 2.0 + rng.standard_normal(1000) * 0.05,
        }))
        members = [m0, m1, m2]
        wf = EnsembleAverageWorkflow(
            column_name="signal",
            trim_method="std",
            window_size=50,
            start_time=0.0,
        )
        result = wf.run(members)
        assert result["interpolation_required"] is True
        assert result["common_grid"] is not None

    def test_keep_intermediate_false_by_default(self, same_grid_members):
        wf = EnsembleAverageWorkflow(column_name="signal", window_size=50)
        result = wf.run(same_grid_members)
        assert result["averaged_stream"] is None

    def test_keep_intermediate_true(self, same_grid_members):
        wf = EnsembleAverageWorkflow(
            column_name="signal", window_size=50, keep_intermediate=True
        )
        result = wf.run(same_grid_members)
        assert isinstance(result["averaged_stream"], DataStream)

    def test_run_accepts_ensemble_object(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        wf = EnsembleAverageWorkflow(column_name="signal", window_size=50)
        result = wf.run(ens)
        assert result["n_members"] == len(same_grid_members)

    def test_verbosity_doesnt_crash(self, same_grid_members, capsys):
        wf = EnsembleAverageWorkflow(
            column_name="signal", window_size=50, verbosity=2
        )
        result = wf.run(same_grid_members)
        captured = capsys.readouterr()
        assert "EnsembleAverageWorkflow" in captured.out

    def test_missing_column_raises(self, same_grid_members):
        wf = EnsembleAverageWorkflow(column_name="nonexistent", window_size=50)
        with pytest.raises(KeyError, match="nonexistent"):
            wf.run(same_grid_members)

    def test_invalid_member_type_raises(self):
        wf = EnsembleAverageWorkflow(column_name="signal", window_size=50)
        with pytest.raises((TypeError, ValueError)):
            wf.run("not_a_list")

    def test_empty_members_raises(self):
        wf = EnsembleAverageWorkflow(column_name="signal", window_size=50)
        with pytest.raises((ValueError, TypeError)):
            wf.run([])

    def test_result_has_metadata(self, same_grid_members):
        wf = EnsembleAverageWorkflow(column_name="signal", window_size=50)
        result = wf.run(same_grid_members)
        assert "metadata" in result
        assert "trim_method" in result["metadata"]

    def test_transient_trimming(self, transient_members):
        wf = EnsembleAverageWorkflow(
            column_name="signal",
            trim_method="std",
            window_size=50,
            start_time=0.0,
        )
        result = wf.run(transient_members)
        # Trimmed should have fewer rows than the full average
        full_len = len(compute_average_ensemble(transient_members).data)
        trimmed_len = len(result["trimmed_stream"].data)
        assert trimmed_len <= full_len


# ---------------------------------------------------------------------------
# Tests: EnsembleStatisticsWorkflow
# ---------------------------------------------------------------------------

class TestEnsembleStatisticsWorkflow:

    def test_import(self):
        from quends.workflow import EnsembleStatisticsWorkflow  # noqa: F401

    def test_top_level_import(self):
        from quends import EnsembleStatisticsWorkflow  # noqa: F401

    def test_invalid_technique_raises(self):
        with pytest.raises(ValueError, match="technique"):
            EnsembleStatisticsWorkflow(technique="bad_technique")

    # ── pooled_block_means (T1) ───────────────────────────────────────

    def test_run_pooled_block_means(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal",
            technique="pooled_block_means",
            trim_method="std",
            window_size=50,
        )
        result = wf.run(same_grid_members)

        assert result["workflow"] == "ensemble_statistics"
        assert result["technique"] == "pooled_block_means"
        assert "pooled_block_means" in result
        assert "ivw_member_means" not in result
        t1 = result["pooled_block_means"]
        assert "statistics" in t1
        assert "signal" in t1["statistics"]["results"]
        assert np.isfinite(t1["statistics"]["results"]["signal"].get("mean", np.nan))

    def test_pooled_block_means_has_metadata(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="pooled_block_means", window_size=50
        )
        result = wf.run(same_grid_members)
        assert (
            "technique_1_pooled_block_means"
            in result["pooled_block_means"]["statistics"]["metadata"]
        )

    # ── Backward-compat: legacy "technique1" string still works ───────

    def test_legacy_technique1_alias(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="technique1", window_size=50
        )
        result = wf.run(same_grid_members)
        # Constructor normalises the technique to its canonical name.
        assert result["technique"] == "pooled_block_means"
        assert "pooled_block_means" in result

    # ── ivw_member_means (T2) ─────────────────────────────────────────

    def test_run_ivw_member_means(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal",
            technique="ivw_member_means",
            trim_method="std",
            window_size=50,
        )
        result = wf.run(same_grid_members)

        assert result["technique"] == "ivw_member_means"
        assert "ivw_member_means" in result
        assert "pooled_block_means" not in result
        t2 = result["ivw_member_means"]
        stats = t2["statistics"]["results"]["signal"]
        assert stats["se_method"] == "ivw_member_means"

    def test_ivw_member_means_full_diagnostics(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal",
            technique="ivw_member_means",
            window_size=50,
            diagnostics="full",
        )
        result = wf.run(same_grid_members)
        stats = result["ivw_member_means"]["statistics"]["results"]["signal"]
        assert stats.get("individual") is not None

    def test_legacy_technique2_alias(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="technique2", window_size=50
        )
        result = wf.run(same_grid_members)
        assert result["technique"] == "ivw_member_means"
        assert "ivw_member_means" in result

    # ── Both ──────────────────────────────────────────────────────────

    def test_run_both(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal",
            technique="both",
            trim_method="std",
            window_size=50,
        )
        result = wf.run(same_grid_members)

        assert "pooled_block_means" in result
        assert "ivw_member_means" in result
        # Both should have valid means
        m1 = result["pooled_block_means"]["statistics"]["results"]["signal"]["mean"]
        m2 = result["ivw_member_means"]["statistics"]["results"]["signal"]["mean"]
        assert np.isfinite(m1)
        assert np.isfinite(m2)

    # ── Edge cases ────────────────────────────────────────────────────

    def test_accepts_ensemble_object(self, same_grid_members):
        ens = Ensemble(same_grid_members)
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="pooled_block_means", window_size=50
        )
        result = wf.run(ens)
        assert result["n_members"] == len(same_grid_members)

    def test_none_column_resolves_all_common(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name=None, technique="pooled_block_means", window_size=50
        )
        result = wf.run(same_grid_members)
        cols = set(result["pooled_block_means"]["statistics"]["results"].keys())
        assert "signal" in cols

    def test_verbosity_doesnt_crash(self, same_grid_members, capsys):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="both", window_size=50, verbosity=1
        )
        wf.run(same_grid_members)
        captured = capsys.readouterr()
        assert "EnsembleStatisticsWorkflow" in captured.out

    def test_missing_column_raises(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="nonexistent",
            technique="pooled_block_means",
            window_size=50,
        )
        with pytest.raises((ValueError, KeyError)):
            wf.run(same_grid_members)

    def test_empty_members_raises(self):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="pooled_block_means", window_size=50
        )
        with pytest.raises((ValueError, TypeError)):
            wf.run([])

    def test_keep_trimmed_false_by_default(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal", technique="pooled_block_means", window_size=50
        )
        result = wf.run(same_grid_members)
        assert result["pooled_block_means"]["trimmed_members"] is None

    def test_keep_trimmed_true(self, same_grid_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal",
            technique="pooled_block_means",
            window_size=50,
            keep_trimmed=True,
        )
        result = wf.run(same_grid_members)
        assert result["pooled_block_means"]["trimmed_members"] is not None

    def test_transient_members(self, transient_members):
        wf = EnsembleStatisticsWorkflow(
            column_name="signal",
            technique="both",
            trim_method="std",
            window_size=50,
        )
        result = wf.run(transient_members)
        assert "pooled_block_means" in result
        assert "ivw_member_means" in result


# ---------------------------------------------------------------------------
# Tests: BatchEnsembleWorkflow (scaffold)
# ---------------------------------------------------------------------------

class TestBatchEnsembleWorkflow:

    def test_import(self):
        from quends.workflow import BatchEnsembleWorkflow  # noqa: F401

    def test_top_level_import(self):
        from quends import BatchEnsembleWorkflow  # noqa: F401

    def test_construction_valid(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            ensemble_groups=[Ensemble(same_grid_members)],
            column_name="signal",
        )
        assert wf.n_groups == 1

    def test_invalid_sub_workflow_type(self, same_grid_members):
        with pytest.raises(ValueError, match="sub_workflow_type"):
            BatchEnsembleWorkflow(
                ensemble_groups=[Ensemble(same_grid_members)],
                sub_workflow_type="unknown",
            )

    def test_invalid_technique(self, same_grid_members):
        with pytest.raises(ValueError, match="technique"):
            BatchEnsembleWorkflow(
                ensemble_groups=[Ensemble(same_grid_members)],
                technique="bad",
            )

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            BatchEnsembleWorkflow(ensemble_groups=[])

    def test_groups_not_list_raises(self):
        with pytest.raises(TypeError):
            BatchEnsembleWorkflow(ensemble_groups="not_a_list")

    def test_none_groups_at_construction(self):
        wf = BatchEnsembleWorkflow(ensemble_groups=None, column_name="signal")
        assert wf.n_groups is None

    def test_run_statistics_default(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            ensemble_groups=[Ensemble(same_grid_members)],
            column_name="signal",
            sub_workflow_type="statistics",
            technique="pooled_block_means",
            batch_config={"window_size": 50},
        )
        result = wf.run()
        assert result["workflow"] == "batch_ensemble"
        assert result["n_items"] == 1
        assert result["n_success"] == 1
        assert result["n_failed"] == 0
        assert "item_0" in result["results"]

    def test_run_with_groups_arg(self, same_grid_members):
        # construction without groups, then pass to run()
        wf = BatchEnsembleWorkflow(
            column_name="signal",
            sub_workflow_type="statistics",
            technique="pooled_block_means",
            batch_config={"window_size": 50},
        )
        result = wf.run(ensemble_groups=[Ensemble(same_grid_members)])
        assert result["n_success"] == 1

    def test_run_average_workflow(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            ensemble_groups=[Ensemble(same_grid_members)],
            column_name="signal",
            sub_workflow_type="average",
            batch_config={"window_size": 50},
        )
        result = wf.run()
        assert result["n_success"] == 1
        per_item = result["results"]["item_0"]
        assert per_item["workflow"] == "ensemble_average"

    def test_run_dict_items_with_id(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            sub_workflow_type="statistics",
            technique="pooled_block_means",
            batch_config={"column_name": "signal", "window_size": 50},
        )
        result = wf.run(
            ensemble_groups=[
                {"name": "alpha", "members": same_grid_members},
                {"id": "beta", "ensemble": Ensemble(same_grid_members)},
            ]
        )
        assert result["n_items"] == 2
        assert result["n_success"] == 2
        assert "alpha" in result["results"]
        assert "beta" in result["results"]

    def test_run_continue_on_error(self, same_grid_members):
        # Mix a valid and an invalid item; with continue_on_error=True we
        # collect errors but do not raise.
        wf = BatchEnsembleWorkflow(
            sub_workflow_type="statistics",
            technique="pooled_block_means",
            batch_config={"column_name": "signal", "window_size": 50},
        )
        result = wf.run(
            ensemble_groups=[
                {"name": "good", "members": same_grid_members},
                # "bad" provides no members at all → ValueError on unpack
                {"name": "bad"},
            ],
            continue_on_error=True,
        )
        assert result["n_success"] == 1
        assert result["n_failed"] == 1
        assert "bad" in result["errors"]

    def test_run_continue_on_error_false_reraises(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            sub_workflow_type="statistics",
            technique="pooled_block_means",
            batch_config={"column_name": "signal", "window_size": 50},
        )
        with pytest.raises(ValueError):
            wf.run(
                ensemble_groups=[{"name": "bad"}],
                continue_on_error=False,
            )

    def test_run_no_groups_raises_value_error(self):
        wf = BatchEnsembleWorkflow(column_name="signal")
        with pytest.raises(ValueError, match="No ensemble_groups"):
            wf.run()

    def test_sub_workflow_type_property(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            ensemble_groups=[Ensemble(same_grid_members)],
            sub_workflow_type="average",
        )
        assert wf.sub_workflow_type == "average"

    def test_technique_property(self, same_grid_members):
        wf = BatchEnsembleWorkflow(
            ensemble_groups=[Ensemble(same_grid_members)],
            technique="pooled_block_means",
        )
        assert wf.technique == "pooled_block_means"

    def test_legacy_technique_alias_normalised(self, same_grid_members):
        # Backward compat: passing "technique1" should still work and be
        # normalised to the canonical name.
        wf = BatchEnsembleWorkflow(
            ensemble_groups=[Ensemble(same_grid_members)],
            technique="technique1",
        )
        assert wf.technique == "pooled_block_means"


# ---------------------------------------------------------------------------
# Import smoke-checks from top-level quends package
# ---------------------------------------------------------------------------

class TestTopLevelImports:
    def test_ensemble_average_workflow_importable(self):
        from quends import EnsembleAverageWorkflow
        assert EnsembleAverageWorkflow is not None

    def test_ensemble_statistics_workflow_importable(self):
        from quends import EnsembleStatisticsWorkflow
        assert EnsembleStatisticsWorkflow is not None

    def test_batch_ensemble_workflow_importable(self):
        from quends import BatchEnsembleWorkflow
        assert BatchEnsembleWorkflow is not None

    def test_robust_workflow_still_importable(self):
        from quends import RobustWorkflow
        assert RobustWorkflow is not None

    def test_ensemble_still_importable(self):
        from quends import Ensemble
        assert Ensemble is not None

    def test_datastream_still_importable(self):
        from quends import DataStream
        assert DataStream is not None

    def test_workflow_module_imports(self):
        from quends.workflow import (
            BatchEnsembleWorkflow,
            EnsembleAverageWorkflow,
            EnsembleStatisticsWorkflow,
            RobustWorkflow,
        )

    def test_ensemble_utils_importable(self):
        from quends.base.ensemble_utils import (
            check_time_steps_uniformity,
            compute_average_ensemble,
            direct_average,
            get_common_variables,
            interpolate_to_common_time,
            resolve_cols,
            trim_members,
            validate_column,
            validate_members,
        )

    def test_ensemble_statistics_importable(self):
        from quends.base.ensemble_statistics import (
            compute_ensemble_statistics,
            ensemble_average_stats_for_col,
            ivw_member_means_stats_for_col,
            pool_block_means,
            pooled_block_means_stats_for_col,
        )


def test_ensemble_statistics_metadata_has_budget_fields(same_grid_members):
    from quends.base.ensemble_statistics import compute_ensemble_statistics
    out = compute_ensemble_statistics(same_grid_members, "signal", technique="pooled_block_means")
    md = out["metadata"]
    assert md["estimator"] == "pooled_block_means"
    assert md["n_members"] == len(same_grid_members)
    assert md["schema_version"] == "1.0"
    assert md["total_samples"] > 0


def test_ensemble_from_files_and_compute_uncertainty(tmp_path):
    import numpy as np
    from quends import Ensemble
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):
        p = tmp_path / f"ens_run_{i:04d}.csv"
        pd.DataFrame({"time": np.arange(800.0),
                      "HeatFlux_st": 5 + 0.2 * rng.standard_normal(800)}).to_csv(p, index=False)
        paths.append(str(p))

    ens = Ensemble.from_files(paths, variable="HeatFlux_st")
    assert len(ens) == 4
    assert list(ens.get_member(0).data.columns) == ["time", "HeatFlux_st"]

    # compute_uncertainty(method=...) == compute_statistics(technique=...)
    a = ens.compute_uncertainty(method="pooled_block_means")
    b = ens.compute_statistics("HeatFlux_st", technique="pooled_block_means")
    assert a["results"]["HeatFlux_st"]["mean_uncertainty"] == \
           b["results"]["HeatFlux_st"]["mean_uncertainty"]
    assert a["metadata"]["n_members"] == 4
