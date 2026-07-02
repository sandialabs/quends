import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import nbformat
import pandas as pd
import pandas.testing as pdt
import papermill as pm
import pytest

pytest_plugins = ("tests._shared",)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "tutorial" / "notebooks"
INPUT_NOTEBOOK = NOTEBOOK_DIR / "start_time_demo.ipynb"
START_TIME_DEMO_DIR = REPO_ROOT / "tests" / "start_time_demo"
OUTPUT_DIR = START_TIME_DEMO_DIR / "output"
EXPECTED_DIR = START_TIME_DEMO_DIR / "expected"


def execute_notebook() -> Path:
    """Execute the notebook and return the path to the executed version."""
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
            cell.get("execution_count") for cell in executed_nb.cells
        ), "No cells executed."

        return output_nb


def load_csv_pair(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both current and expected CSV files."""
    current_path = OUTPUT_DIR / filename
    expected_path = EXPECTED_DIR / filename

    if not current_path.exists():
        raise ValueError(f"Output CSV not found at {current_path}")
    if not expected_path.exists():
        raise ValueError(f"Expected CSV not found at {expected_path}")

    return pd.read_csv(current_path), pd.read_csv(expected_path)


def compare_results(filename: str, atol: float = 1e-8):
    """Compare notebook output against expected baseline."""
    current, expected = load_csv_pair(filename)
    shared_cols = [c for c in expected.columns if c in current.columns]
    assert shared_cols, f"No shared columns found for {filename}"
    pdt.assert_frame_equal(
        current[shared_cols], expected[shared_cols], atol=atol, check_dtype=False
    )


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Execute notebook once before running all start-time demo tests."""
    execute_notebook()


def test_flat_signal_with_time_of_restart():
    compare_results("flat_signal_stats_with_time_of_restart.csv")


def test_flat_signal_without_time_of_restart():
    compare_results("flat_signal_stats_without_time_of_restart.csv")


def test_flat_signal_with_noise():
    compare_results("flat_signal_stats_with_noise.csv")
