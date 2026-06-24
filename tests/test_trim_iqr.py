"""Tests for IQRTrimStrategy (the ``iqr`` trim method)."""

import numpy as np
import pandas as pd

from quends import DataStream
from quends.base.trim import (
    IQRTrimStrategy,
    TrimDataStreamOperation,
    build_trim_strategy,
)


def _transient_then_steady(n_transient=80, n_steady=320, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_transient + n_steady)
    signal = np.concatenate(
        [np.linspace(0.0, 10.0, n_transient), 10.0 + rng.normal(0.0, noise, n_steady)]
    )
    return DataStream(pd.DataFrame({"time": t, "signal": signal}))


def test_iqr_detects_steady_state():
    ds = _transient_then_steady()
    # A generous threshold lets the IQR criterion clear once the signal settles.
    op = TrimDataStreamOperation(strategy=IQRTrimStrategy(window_size=40, threshold=0.3))
    result = op(ds, column_name="signal")

    assert isinstance(result, DataStream)
    assert "sss_start" in result.trim_metadata
    assert 0 < len(result) <= len(ds)
    assert result.trim_metadata["sss_start"] is not None


def test_iqr_method_name():
    assert IQRTrimStrategy(window_size=20).method_name == "iqr"


def test_iqr_via_factory_handles_no_detection():
    # A strict threshold the IQR never clears -> the no-detection path.
    ds = _transient_then_steady()
    op = TrimDataStreamOperation(
        strategy=build_trim_strategy(method="iqr", window_size=40, threshold=1e-6)
    )
    result = op(ds, column_name="signal")
    assert isinstance(result, DataStream)
