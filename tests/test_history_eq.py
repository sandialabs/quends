"""Tests for DataStreamHistory length, indexing, and equality semantics."""

from quends.base.history import DataStreamHistory, DataStreamHistoryEntry


def _history():
    return DataStreamHistory(
        [
            DataStreamHistoryEntry("load", {"a": 1}),
            DataStreamHistoryEntry("trim", {"w": 50}),
        ]
    )


def test_len_and_getitem():
    h = _history()
    assert len(h) == 2
    assert h[0] == {"operation": "load", "options": {"a": 1}}
    assert h[1] == {"operation": "trim", "options": {"w": 50}}


def test_eq_other_history():
    h = _history()
    assert h == _history()
    assert h != DataStreamHistory([DataStreamHistoryEntry("load", {"a": 1})])


def test_eq_list_of_dicts_matching():
    h = _history()
    assert h == [
        {"operation": "load", "options": {"a": 1}},
        {"operation": "trim", "options": {"w": 50}},
    ]


def test_eq_list_length_mismatch():
    h = _history()
    assert (h == [{"operation": "load", "options": {"a": 1}}]) is False


def test_eq_list_with_non_dict_element():
    h = _history()
    assert (h == ["not-a-dict", "also-not"]) is False


def test_eq_list_operation_mismatch():
    h = _history()
    assert (
        h
        == [
            {"operation": "WRONG", "options": {"a": 1}},
            {"operation": "trim", "options": {"w": 50}},
        ]
    ) is False


def test_eq_list_options_mismatch():
    h = _history()
    assert (
        h
        == [
            {"operation": "load", "options": {"a": 999}},
            {"operation": "trim", "options": {"w": 50}},
        ]
    ) is False


def test_eq_unsupported_type_is_not_equal():
    h = _history()
    # __eq__ returns NotImplemented -> Python falls back to identity -> False.
    assert (h == 5) is False
