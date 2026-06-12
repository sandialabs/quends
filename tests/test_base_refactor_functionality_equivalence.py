import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quends.base import (
    DataStream,
    Ensemble,
    MakeDataStreamStationaryOperation,
    NoiseThresholdTrimStrategy,
    QuantileTrimStrategy,
    RollingVarianceThresholdTrimStrategy,
    TrimDataStreamOperation,
    build_trim_strategy,
)
from quends.base.history import DataStreamHistory


def transient_to_stationary_stream() -> DataStream:
    rng = np.random.default_rng(123)
    time = np.arange(1000, dtype=float)
    signal = np.concatenate(
        [np.linspace(0.0, 5.0, 200), 5.0 + 0.1 * rng.standard_normal(800)]
    )
    return DataStream(pd.DataFrame({"time": time, "phi2": signal}))


def shifted_stream(shift: float = 0.05) -> DataStream:
    ds = transient_to_stationary_stream()
    df = ds.data.copy()
    df["time"] = df["time"] + shift
    df["phi2"] = df["phi2"] + 0.01
    return DataStream(df)


def test_base_public_imports_and_datastream_alias():
    ds = transient_to_stationary_stream()

    assert isinstance(ds, DataStream)
    assert ds.df is ds.data
    assert "phi2" in ds.variables()
    assert inspect.isclass(QuantileTrimStrategy)
    assert inspect.isclass(NoiseThresholdTrimStrategy)
    assert inspect.isclass(RollingVarianceThresholdTrimStrategy)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("std", {}),
        ("threshold", {"threshold": 0.03}),
        ("rolling_variance", {"threshold": 1.0}),
    ],
)
def test_old_trim_methods_map_to_refactored_strategies(method, kwargs):
    ds = transient_to_stationary_stream()
    strategy = build_trim_strategy(method, window_size=25, **kwargs)
    trimmed = TrimDataStreamOperation(strategy=strategy)(ds, column_name="phi2")

    assert isinstance(trimmed, DataStream)
    assert not trimmed.data.empty
    assert list(trimmed.data.columns) == ["time", "phi2"]
    assert trimmed.data["time"].iloc[0] >= 0.0
    assert isinstance(trimmed.history, DataStreamHistory)
    assert trimmed.history.entries()[-1].operation_name == "trim"
    assert trimmed.history.entries()[-1].parameters["method"] == method


def test_trim_handles_short_series_safely():
    ds = DataStream(pd.DataFrame({"time": np.arange(5), "phi2": np.ones(5)}))
    strategy = QuantileTrimStrategy(window_size=10)

    trimmed = TrimDataStreamOperation(strategy=strategy)(ds, column_name="phi2")

    assert isinstance(trimmed, DataStream)
    assert trimmed.data.empty
    assert hasattr(trimmed, "message")


def test_trim_missing_column_fails_clearly_with_empty_stream():
    ds = transient_to_stationary_stream()
    strategy = QuantileTrimStrategy(window_size=10)

    trimmed = TrimDataStreamOperation(strategy=strategy)(ds, column_name="missing")

    assert isinstance(trimmed, DataStream)
    assert trimmed.data.empty
    assert "missing" in trimmed.message


def test_stationarity_schema_and_make_stationary_operation():
    ds = transient_to_stationary_stream()

    stationarity = ds.is_stationary("phi2")
    assert set(stationarity) == {"phi2"}
    assert isinstance(stationarity["phi2"], bool)

    op = MakeDataStreamStationaryOperation(column="phi2", n_pts_orig=len(ds.data))
    result_ds, stationary = op(ds)

    assert isinstance(result_ds, DataStream)
    assert isinstance(stationary, bool)
    assert result_ds.history.entries()[-1].operation_name == "make_stationary"


def test_statistics_and_effective_sample_size_outputs():
    ds = transient_to_stationary_stream()

    stats = ds.compute_statistics("phi2", window_size=50)
    assert "phi2" in stats
    assert {"mean", "mean_uncertainty", "confidence_interval", "ess_blocks"}.issubset(
        stats["phi2"]
    )

    ess = ds.effective_sample_size("phi2")
    assert "results" in ess
    assert "phi2" in ess["results"]

    block_ess = ds.get_block_effective_n("phi2", window_size=50)
    assert block_ess["effective_n"] >= 1.0


def test_ensemble_statistics_and_average_paths():
    ds1 = transient_to_stationary_stream()
    ds2 = shifted_stream()
    ensemble = Ensemble([ds1, ds2])

    stats = ensemble.compute_statistics("phi2", technique=1, window_size=50)
    assert "results" in stats
    assert "phi2" in stats["results"]

    averaged = ensemble.compute_average_ensemble(interp_method="linear")
    assert isinstance(averaged, DataStream)
    assert "phi2" in averaged.data.columns

    stationary = ensemble.is_stationary("phi2")
    assert set(stationary) == {"results", "metadata"}
    assert "Member 0" in stationary["results"]


def test_nan_series_statistics_fail_clearly_or_return_nan():
    ds = DataStream(pd.DataFrame({"time": np.arange(10), "phi2": [np.nan] * 10}))

    stats = ds.compute_statistics("phi2", window_size=3)
    assert "error" in stats["phi2"]

    ess = ds.effective_sample_size("phi2")
    assert ess["results"]["phi2"]["effective_sample_size"] is None


def test_base_code_does_not_reference_legacy_install_tree():
    base_dir = Path(__file__).resolve().parents[1] / "src" / "quends" / "base"
    legacy_path_token = "/" + "QQ" + "_install/"
    offenders = []
    for path in base_dir.glob("*.py"):
        if legacy_path_token in path.read_text():
            offenders.append(path.name)
    assert offenders == []
