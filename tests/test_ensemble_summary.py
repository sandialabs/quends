"""Tests for Ensemble accessors and summary helpers."""

import pandas as pd

from quends import DataStream, Ensemble


def _two_member_ensemble():
    m0 = DataStream(pd.DataFrame({"time": [0, 1, 2, 3], "Q": [1.0, 2.0, 3.0, 4.0]}))
    m1 = DataStream(pd.DataFrame({"time": [0, 1, 2, 3], "Q": [2.0, 3.0, 4.0, 5.0]}))
    return Ensemble([m0, m1]), m0, m1


def test_ensemble_accessors():
    ens, m0, m1 = _two_member_ensemble()
    assert ens.get_member(0) is m0
    assert ens.members() == [m0, m1]
    assert ens.common_variables() == ["Q"]

    heads = ens.head(2)
    assert set(heads.keys()) == {0, 1}
    assert len(heads[0]) == 2


def test_ensemble_summary():
    ens, _, _ = _two_member_ensemble()
    summary = ens.summary()
    assert summary["n_members"] == 2
    assert summary["common_variables"] == ["Q"]
    assert set(summary["members"].keys()) == {"Member 0", "Member 1"}
    assert summary["members"]["Member 0"]["n_samples"] == 4


def test_ensemble_summary_verbose(capsys):
    ens, _, _ = _two_member_ensemble()
    summary = ens.summary(verbose=True)
    out = capsys.readouterr().out
    assert summary["n_members"] == 2
    assert "Ensemble members: 2" in out
    assert "Common variables:" in out
