"""Tests for the sign-aware near-zero warm-up stripping in TrimStrategy._preprocess."""

import numpy as np
import pandas as pd

from quends import DataStream
from quends.base.trim import QuantileTrimStrategy


def _stream(values):
    return DataStream(
        pd.DataFrame({"time": np.arange(len(values), dtype=float), "x": values})
    )


def _steady(level=10.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return level + rng.normal(0.0, 0.5, n)


WARMUP = np.full(40, 1e-6)  # near-zero (tiny positive) warm-up plateau


def test_all_positive_warmup_is_stripped():
    strat = QuantileTrimStrategy(window_size=50)
    out = strat._preprocess(_stream(np.r_[WARMUP, _steady(10.0)]), "x")
    # The 40-sample near-zero warm-up is dropped; the kept data starts near 10.
    assert len(out) == 300
    assert out["x"].iloc[0] > 5.0


def test_all_negative_warmup_is_stripped():
    strat = QuantileTrimStrategy(window_size=50)
    out = strat._preprocess(_stream(np.r_[-WARMUP, -_steady(10.0)]), "x")
    assert len(out) == 300
    assert out["x"].iloc[0] < -5.0


def test_mixed_sign_trace_is_not_stripped():
    rng = np.random.default_rng(1)
    mixed = np.r_[rng.normal(0.0, 1.0, 40), _steady(10.0)]
    strat = QuantileTrimStrategy(window_size=50)
    out = strat._preprocess(_stream(mixed), "x")
    # A sign-changing early excursion is informative -> keep everything.
    assert len(out) == len(mixed)


def test_opt_out_disables_stripping():
    strat = QuantileTrimStrategy(window_size=50)
    strat.drop_leading_nonpositive = False
    out = strat._preprocess(_stream(np.r_[WARMUP, _steady(10.0)]), "x")
    assert len(out) == 340  # nothing stripped


def test_no_warmup_no_change():
    # A trace that is "on" from the start is untouched.
    strat = QuantileTrimStrategy(window_size=50)
    steady = _steady(10.0)
    out = strat._preprocess(_stream(steady), "x")
    assert len(out) == len(steady)
