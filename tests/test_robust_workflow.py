import os
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import nbformat
import pandas as pd
import pandas.testing as pdt
import papermill as pm

os.chdir("examples/notebooks")

# Constants
INPUT_NOTEBOOK = Path("robust_workflow.ipynb")
OUTPUT_DIR = Path("../../tests/output")
EXPECTED_DIR = Path("../../tests/expected")


def execute_notebook() -> Path:
    """Execute the notebook and return the path to the executed version."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="papermill.translators"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_nb = Path(tmpdirname) / "executed_notebook.ipynb"
        pm.execute_notebook(str(INPUT_NOTEBOOK), str(output_nb), kernel_name="python3")

        # Verify execution
        assert output_nb.exists(), f"Executed notebook not created at {output_nb}"

        executed_nb = nbformat.read(output_nb, as_version=4)
        assert (
            "papermill" in executed_nb.metadata
        ), "Notebook metadata missing Papermill info."
        assert any(
            cell.get("execution_count") for cell in executed_nb.cells
        ), "No cells executed."

        print("Papermill execution verified successfully.")
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
    """Execute notebook and compare results against expected baseline."""
    execute_notebook()
    current, expected = load_csv_pair(filename)
    pdt.assert_frame_equal(current, expected, atol=atol, check_dtype=False)
    print(
        f"Regression test passed for {filename}: current results match expected baseline."
    )


def test_linear_transient_to_plateau():
    compare_results("linear_transient_to_plateau.csv", atol=1e-8)


def test_slope_to_sine_regression():
    compare_results("slope_to_sine_stats.csv", atol=1e-6)
