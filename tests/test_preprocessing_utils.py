"""Tests for the shared single-variable preprocessing helpers (_utils)."""

import pandas as pd

from quends import DataStream
from quends.preprocessing._utils import (
    _time_resolution_method,
    build_single_variable_frame,
    load_single_variable,
    resolve_time_column,
)


def test_resolve_time_column_variable_is_time_alias():
    df = pd.DataFrame({"time": [0, 1, 2], "Q": [1, 2, 3]})
    assert resolve_time_column(df, "time") is None


def test_resolve_time_column_no_other_columns():
    df = pd.DataFrame({"Q": [1, 2, 3]})
    assert resolve_time_column(df, "Q") is None


def test_resolve_time_column_name_alias():
    df = pd.DataFrame({"t": [0, 1, 2], "Q": [3, 1, 2]})
    assert resolve_time_column(df, "Q") == "t"


def test_resolve_time_column_single_monotonic_candidate():
    # "x" is not a time alias but is monotonic -> resolved by content.
    df = pd.DataFrame({"signal": [3.0, 1.0, 2.0, 5.0], "x": [0, 1, 2, 3]})
    assert resolve_time_column(df, "signal") == "x"


def test_resolve_time_column_prefers_evenly_spaced_candidate():
    # Two monotonic candidates -> the most evenly spaced (xa) wins.
    df = pd.DataFrame({"signal": [3, 1, 2, 5], "xa": [0, 1, 2, 3], "xb": [0, 1, 3, 7]})
    assert resolve_time_column(df, "signal") == "xa"


def test_resolve_time_column_none_when_no_monotonic_axis():
    df = pd.DataFrame({"a": [3, 1, 2], "b": [5, 4, 6]})
    assert resolve_time_column(df, "a") is None


def test_time_resolution_method_variants():
    assert _time_resolution_method(None, "Q", None) == "none"
    assert _time_resolution_method(None, "Q", "time") == "name_alias"
    assert _time_resolution_method(None, "Q", "x") == "monotonic"


def test_build_single_variable_frame_renames_alias_to_time():
    df = pd.DataFrame({"t": [0, 1, 2], "Q": [1.0, 2.0, 3.0]})
    out = build_single_variable_frame(df, "Q")
    assert list(out.columns) == ["time", "Q"]


def test_build_single_variable_frame_without_time():
    df = pd.DataFrame({"Q": [3.0, 1.0, 2.0]})
    out = build_single_variable_frame(df, "Q")
    assert list(out.columns) == ["Q"]


def test_load_single_variable_records_history():
    df = pd.DataFrame({"time": [0, 1, 2], "Q": [1.0, 2.0, 3.0]})
    ds = load_single_variable(df, "Q", source="mem.csv", loader="from_csv")
    assert isinstance(ds, DataStream)
    assert list(ds.data.columns) == ["time", "Q"]
    entry = ds.history[0]
    assert entry["operation"] == "load"
    assert entry["options"]["variable"] == "Q"
    assert entry["options"]["time_column"] == "time"
    assert entry["options"]["time_resolution"] == "name_alias"
