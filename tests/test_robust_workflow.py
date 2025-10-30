import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import papermill as pm


def test_slope_to_sine_regression():
    """
    Runs the Robust Workflow notebook, generates output stats CSV,
    and compares it to the stored expected baseline to our slope to sine csv.
    """
    # Suppress warnings from Papermill
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="papermill.translators"
    )

    # Define notebook paths
    input_nb = Path("examples/notebooks/robust_workflow.ipynb")
    test_data_path = Path("examples/notebooks/cgyro/output_nu0_02.csv").resolve()

    # Verify the test data exists before running
    if not test_data_path.exists():
        raise ValueError(f"Test data not found at {test_data_path}")

    # Create a temporary directory for the executed notebook
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_nb = Path(tmpdirname) / "executed_notebook.ipynb"

        # Execute the notebook with Papermill
        pm.execute_notebook(str(input_nb), str(output_nb), kernel_name="python3")

        # Load the newly generated stats CSV from the notebook
        current_csv_path = Path("tests/output/slope_to_sine_stats.csv")
        if not current_csv_path.exists():
            raise ValueError(f"Output CSV not found at {current_csv_path}")

        current = pd.read_csv(current_csv_path)

        # Load the expected baseline CSV
        expected_csv_path = Path("tests/expected/slope_to_sine_stats.csv")
        if not expected_csv_path.exists():
            raise ValueError(f"Expected CSV not found at {expected_csv_path}")

        expected = pd.read_csv(expected_csv_path)

        # Compare the two DataFrames
        pdt.assert_frame_equal(current, expected, atol=1e-6, check_dtype=False)

        print("Regression test passed: current results match expected baseline.")
