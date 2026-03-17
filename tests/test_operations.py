import pandas as pd

from quends import DataStream, DataStreamOperation
from quends.base.history import DataStreamHistoryEntry


class DummyOperation(DataStreamOperation):
    def _apply(self, data_stream: DataStream, **kwargs):
        return DataStream(data_stream.data.copy(), history=data_stream.history)


def test_operation_uses_default_name():
    operation = DummyOperation()
    assert operation.name == "DummyOperation"


def test_operation_appends_history_entry():
    ds = DataStream(pd.DataFrame({"A": [1, 2, 3]}))
    operation = DummyOperation(operation_name="dummy", source="unit-test")

    result = operation(ds, column_name="A")
    entries = result.history.entries()

    assert len(entries) == 1
    assert entries[0] == DataStreamHistoryEntry(
        operation_name="dummy",
        parameters={"source": "unit-test", "column_name": "A"},
    )
