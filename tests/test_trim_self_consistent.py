"""Tests for SelfConsistentTrimStrategy (the ``self_consistent`` trim method)."""

import numpy as np
import pandas as pd

from quends import DataStream
from quends.base.trim import (
    SelfConsistentTrimStrategy,
    TrimDataStreamOperation,
    build_trim_strategy,
)


def _transient_then_steady(n_transient=80, n_steady=320, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_transient + n_steady)
    signal = np.concatenate(
        [np.linspace(0.0, 10.0, n_transient), 10.0 + rng.normal(0.0, 0.5, n_steady)]
    )
    return DataStream(pd.DataFrame({"time": t, "signal": signal}))


def test_self_consistent_detects_steady_state():
    ds = _transient_then_steady()
    op = TrimDataStreamOperation(strategy=SelfConsistentTrimStrategy(window_size=40))
    result = op(ds, column_name="signal")

    assert isinstance(result, DataStream)
    assert "sss_start" in result.trim_metadata
    assert 0 < len(result) <= len(ds)
    assert result.trim_metadata["sss_start"] is not None


def test_self_consistent_method_name():
    assert SelfConsistentTrimStrategy(window_size=20).method_name == "self_consistent"


def test_self_consistent_via_factory_handles_no_detection():
    # A short, monotonic series gives no self-consistent steady-state segment;
    # the strategy should still return a DataStream (the no-detection path).
    ds = DataStream(pd.DataFrame({"time": list(range(10)), "signal": list(range(10))}))
    op = TrimDataStreamOperation(
        strategy=build_trim_strategy(method="self_consistent", window_size=20)
    )
    result = op(ds, column_name="signal")
    assert isinstance(result, DataStream)


def _detect(strategy, df, col="signal"):
    return strategy._detection_method(df, col)


def test_self_consistent_detection_guards_return_none():
    df = pd.DataFrame({"time": np.arange(10.0), "signal": np.ones(10)})
    s = SelfConsistentTrimStrategy(window_size=4)
    assert _detect(s, pd.DataFrame()) is None
    assert _detect(s, None) is None
    assert _detect(s, df.drop(columns="time")) is None
    assert _detect(s, df, col="missing") is None
    assert _detect(SelfConsistentTrimStrategy(window_size=0), df) is None
    assert _detect(SelfConsistentTrimStrategy(window_size=6), df) is None  # n < 2W


def test_self_consistent_non_robust_detects_steady_state():
    ds = _transient_then_steady()
    s = SelfConsistentTrimStrategy(window_size=40, robust=False)
    t0 = _detect(s, ds.data)
    assert t0 is not None
    assert t0 >= 40


def test_self_consistent_robust_falls_back_to_std_when_mad_is_zero():
    # Most samples identical -> MAD == 0, so the std fallback is used.
    x = np.full(40, 5.0)
    x[::7] = 5.1
    df = pd.DataFrame({"time": np.arange(40.0), "signal": x})
    s = SelfConsistentTrimStrategy(window_size=10, rel_tol_mu=1.0, rel_tol_sigma=10.0)
    assert _detect(s, df) is not None


def test_self_consistent_non_finite_blocks_are_rejected():
    x = np.full(40, np.nan)
    df = pd.DataFrame({"time": np.arange(40.0), "signal": x})
    assert _detect(SelfConsistentTrimStrategy(window_size=10), df) is None
