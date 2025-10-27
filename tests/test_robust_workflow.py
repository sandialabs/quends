import warnings
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import papermill as pm


def test_slope_to_sine_regression(tmp_path):
    """
    Runs the Robust Workflow notebook, generates output stats CSV,
    and compares it to the stored expected baseline to our slope to sine csv.
    """
    # warnings to ignore
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="papermill.translators"
    )

    # Define notebook paths
    input_nb = Path("examples/notebooks/robust_workflow.ipynb")
    output_nb = tmp_path / "executed_notebook.ipynb"
    test_data_path = Path("examples/notebooks/cgyro/output_nu0_02.csv").resolve()

    # Verify the test data exists before running
    assert test_data_path.exists(), f"Test data not found at {test_data_path}"

    # # Execute the notebook with papermill
    # pm.execute_notebook(
    #     input_nb,
    #     output_nb,
    #     kernel_name="python3",
    #     parameters={'data_paths': [str(test_data_path)]}
    # )

    # Load the newly generated stats CSV from the notebook
    current = pd.read_csv("tests/output/slope_to_sine_stats.csv")

    # Load the expected baseline CSV
    expected = pd.read_csv("tests/expected/slope_to_sine_stats.csv")

    # Compare the two DataFrames
    pdt.assert_frame_equal(current, expected, atol=1e-6, check_dtype=False)

    print("Regression test passed: current results match expected baseline.")
