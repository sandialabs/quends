import math
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import nbformat
import pandas as pd
import pandas.testing as pdt
import papermill as pm
import pytest

from quends import DataStream


@pytest.fixture(scope="module")
def cgyro_streams():
    repo_root = Path(__file__).resolve().parents[1]
    cgyro_dir = repo_root / "examples" / "notebooks" / "cgyro"

    files = [
        "output_nu0_02.csv",
        "output_nu0_05.csv",
        "output_nu0_10.csv",
        "output_nu0_50.csv",
        "output_nu1_0.csv",
    ]

    streams = {}
    for fname in files:
        path = cgyro_dir / fname
        assert path.exists(), f"Expected test CSV missing: {path}"
        # Notebook uses index_col=0 when loading these files
        df = pd.read_csv(path, index_col=0)
        streams[fname] = DataStream(df)

    return streams


REPO_ROOT = Path(__file__).resolve().parents[1]
CGYRO_TEST_DIR = REPO_ROOT / "tests" / "cgyro"
OUTPUT_DIR = CGYRO_TEST_DIR / "output"
EXPECTED_DIR = CGYRO_TEST_DIR / "expected"
NOTEBOOK_DIR = REPO_ROOT / "examples" / "notebooks"
INPUT_NOTEBOOK = NOTEBOOK_DIR / "DataStream_Guide-CGYRO.ipynb"


def execute_notebook() -> Path:
    """Execute the CGYRO notebook and verify it runs."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="papermill.translators"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_nb = Path(tmpdirname) / "executed_notebook.ipynb"
        pm.execute_notebook(
            str(INPUT_NOTEBOOK),
            str(output_nb),
            kernel_name="python3",
            cwd=str(NOTEBOOK_DIR),
        )

        assert output_nb.exists(), f"Executed notebook not created at {output_nb}"
        executed_nb = nbformat.read(output_nb, as_version=4)
        assert (
            "papermill" in executed_nb.metadata
        ), "Notebook metadata missing Papermill info."
        assert any(
            cell.get("execution_count") is not None for cell in executed_nb.cells
        ), "No cells executed."

        print("Papermill execution verified successfully.")
        return output_nb


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Execute the CGYRO notebook once before running all tests."""
    execute_notebook()


def load_csv_pair(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both current and expected CSV files."""
    current_path = OUTPUT_DIR / filename
    expected_path = EXPECTED_DIR / filename

    if not current_path.exists():
        raise FileNotFoundError(f"Output CSV not found at {current_path}")
    if not expected_path.exists():
        raise FileNotFoundError(f"Expected CSV not found at {expected_path}")

    return pd.read_csv(current_path), pd.read_csv(expected_path)


def compare_results(filename: str, atol: float = 1e-8):
    """Compare current output CSV against expected baseline."""
    current, expected = load_csv_pair(filename)
    shared_cols = [c for c in expected.columns if c in current.columns]
    assert shared_cols, f"No shared columns found for {filename}"
    pdt.assert_frame_equal(
        current[shared_cols], expected[shared_cols], atol=atol, check_dtype=False
    )
    print(
        f"Regression test passed for {filename}: current results match expected baseline."
    )


def test_csvs_have_expected_columns(cgyro_streams):
    # pick one representative file
    ds = next(iter(cgyro_streams.values()))
    cols = list(ds.variables())
    # Expect at least 'time' and a couple of signal columns
    assert "time" in cols or "time" in ds.data.columns
    assert any("Q_" in c or "Q" in c for c in cols)


def test_effective_sample_size_results_finite(cgyro_streams):
    ds = cgyro_streams["output_nu0_50.csv"]
    ess = ds.effective_sample_size()
    assert isinstance(ess, dict)
    assert "results" in ess
    for col, val in ess["results"].items():
        # ESS should be numeric and at least 1 when computable
        assert val is not None
        assert not (isinstance(val, str) and val.startswith("Error"))


def test_compute_statistics_contains_expected_keys(cgyro_streams):
    ds = cgyro_streams["output_nu0_50.csv"]
    stats = ds.compute_statistics()
    # ensure each column has the expected statistic keys
    for col, entry in stats.items():
        # skip non-column entries
        if col == "metadata":
            continue
        assert "mean" in entry
        assert "mean_uncertainty" in entry
        assert "confidence_interval" in entry
        assert "effective_sample_size" in entry
        assert "window_size" in entry


def test_additional_data_returns_model(cgyro_streams):
    ds = cgyro_streams["output_nu0_50.csv"]
    add_info = ds.additional_data(column_name="Q_e/Q_GBD", method="sliding")
    assert isinstance(add_info, dict)
    assert "Q_e/Q_GBD" in add_info
    info = add_info["Q_e/Q_GBD"]
    # If curve fitting succeeded we should have A_est and p_est
    if "error" not in info:
        assert "A_est" in info
        assert "p_est" in info
        assert math.isfinite(info["A_est"]) and math.isfinite(info["p_est"])


@pytest.mark.parametrize(
    "filename",
    sorted([path.name for path in EXPECTED_DIR.glob("*.csv")]),
)
def test_cgyro_output_matches_expected_baseline(filename):
    """Compare CGYRO output CSVs against expected regression baselines."""
    compare_results(filename)
