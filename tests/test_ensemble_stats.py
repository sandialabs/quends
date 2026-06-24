"""Tests for Ensemble per-statistic accessors across the three techniques."""

import numpy as np
import pandas as pd
import pytest

from quends import DataStream, Ensemble

COL = "x"


def _ensemble(n_members=3, n=300, seed=1):
    rng = np.random.default_rng(seed)
    return Ensemble(
        [
            DataStream(
                pd.DataFrame(
                    {"time": np.arange(float(n)), COL: 5.0 + 0.2 * rng.standard_normal(n)}
                )
            )
            for _ in range(n_members)
        ]
    )


def test_is_stationary_report():
    rep = _ensemble().is_stationary(COL)
    assert set(rep.keys()) == {"results", "metadata"}
    assert "Member 0" in rep["results"]


@pytest.mark.parametrize(
    "technique", ["ensemble_average", "pooled_block_means", "ivw"]
)
def test_mean_sem_ci_each_technique(technique):
    ens = _ensemble()
    m = ens.mean(column_name=COL, technique=technique)
    u = ens.mean_uncertainty(column_name=COL, technique=technique)
    ci = ens.confidence_interval(column_name=COL, technique=technique)
    for out in (m, u, ci):
        assert "results" in out and COL in out["results"]
    assert np.isfinite(m["results"][COL])


def test_ess_blocks_and_n_short_averages():
    ens = _ensemble()
    for out in (
        ens.effective_sample_size(COL),
        ens.effective_sample_size_blocks(COL),
        ens.n_short_averages(COL),
    ):
        assert COL in out["results"]
