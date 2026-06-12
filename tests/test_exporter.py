# Tests for the Exporter (overwrite protection, roundtrip, native types, sidecar).
import json

import numpy as np
import pandas as pd
import pytest

from quends import Exporter


def test_save_dataframe_roundtrip_and_returns_path(tmp_path):
    exp = Exporter(output_dir=str(tmp_path))
    df = pd.DataFrame({"time": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
    path = exp.save_dataframe(df, "out.csv")
    assert path.endswith("out.csv")
    pd.testing.assert_frame_equal(pd.read_csv(path), df)


def test_overwrite_guard(tmp_path):
    exp = Exporter(output_dir=str(tmp_path))
    df = pd.DataFrame({"x": [1, 2]})
    exp.save_dataframe(df, "a.csv")
    with pytest.raises(FileExistsError):
        exp.save_dataframe(df, "a.csv")
    # explicit overwrite is allowed
    assert exp.save_dataframe(df, "a.csv", overwrite=True).endswith("a.csv")


def test_overwrite_true_constructor(tmp_path):
    exp = Exporter(output_dir=str(tmp_path), overwrite=True)
    df = pd.DataFrame({"x": [1]})
    exp.save_dataframe(df, "b.csv")
    # no raise on second save
    exp.save_dataframe(df, "b.csv")


def test_to_native_types_is_json_serializable():
    exp = Exporter(output_dir=".")
    payload = {"a": np.float64(1.5), "b": np.int64(3), "c": (np.float32(2.0), 4)}
    native = exp.to_native_types(payload)
    json.dumps(native)  # must not raise


def test_save_results_writes_sidecar(tmp_path):
    exp = Exporter(output_dir=str(tmp_path))
    results = {"mean": np.float64(5.5), "sem": np.float64(0.02)}
    data_path, meta_path = exp.save_results(
        results, "run1", metadata={"source": "ens_run_0001.csv", "variable": "HeatFlux_st"}
    )
    assert meta_path.endswith("run1.meta.json")
    meta = json.load(open(meta_path))
    assert meta["schema_version"] == "1.0"
    assert meta["variable"] == "HeatFlux_st"
    assert meta["data_file"] == "run1.json"
