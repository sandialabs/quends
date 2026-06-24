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
