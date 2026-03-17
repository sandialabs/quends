from quends.base.history import DataStreamHistory, DataStreamHistoryEntry


def test_history_entry_is_immutable():
    entry = DataStreamHistoryEntry(operation_name="trim", parameters={"column": "A"})
    assert entry.operation_name == "trim"
    assert entry.parameters == {"column": "A"}


def test_history_append_returns_self_and_stores_entries():
    first = DataStreamHistoryEntry(operation_name="mean", parameters={"window_size": 1})
    second = DataStreamHistoryEntry(operation_name="trim", parameters={"column": "A"})

    history = DataStreamHistory()
    returned = history.append(first).append(second)

    assert returned is history
    assert history.entries() == (first, second)
