import tempfile
import warnings
from pathlib import Path

import nbformat
import papermill as pm

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "notebooks"
INPUT_NOTEBOOK = NOTEBOOK_DIR / "tutorial.ipynb"


def test_tutorial_notebook_executes():
    """Verify that the tutorial notebook executes from start to finish."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="papermill.translators"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_nb = Path(tmpdirname) / "executed_tutorial.ipynb"

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
        ), "Notebook metadata missing Papermill information."
        assert any(
            cell.get("execution_count") for cell in executed_nb.cells
        ), "Notebook contains no executed code cells."
