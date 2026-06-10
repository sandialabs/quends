import re
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import nbformat
import pandas as pd
import pandas.testing as pdt
import papermill as pm
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "examples" / "notebooks"
INPUT_NOTEBOOK = NOTEBOOK_DIR / "DataStream_Guide.ipynb"
GUIDE_TEST_DIR = REPO_ROOT / "tests" / "guide"
OUTPUT_DIR = GUIDE_TEST_DIR / "output"
EXPECTED_DIR = GUIDE_TEST_DIR / "expected"


def notebook_csv_filenames() -> list[str]:
    """Return active CSV baselines written by the notebook."""
    notebook = nbformat.read(INPUT_NOTEBOOK, as_version=4)
    filenames = set()

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if "def save_stats_to_csv" in source:
            continue

        active_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for match in re.finditer(
            r"save_stats_to_csv\s*\(.*?['\"]([^'\"]+\.csv)['\"]",
            active_source,
            flags=re.DOTALL,
        ):
            filenames.add(match.group(1))

    return sorted(filenames)


GUIDE_CSV_FILES = notebook_csv_filenames()


def execute_notebook() -> Path:
    """Execute the DataStream guide notebook and verify it writes output CSVs."""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="papermill.translators"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.csv"):
        path.unlink()

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

        return output_nb


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Execute the guide notebook once before comparing CSV baselines."""
    assert GUIDE_CSV_FILES, "No guide CSV outputs found in DataStream_Guide.ipynb"
    execute_notebook()


def load_csv_pair(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both current and expected CSV files."""
    current_path = OUTPUT_DIR / filename
    expected_path = EXPECTED_DIR / filename

    if not current_path.exists():
        raise FileNotFoundError(f"Output CSV not found at {current_path}")
    if not expected_path.exists():
        raise FileNotFoundError(f"Expected CSV not found at {expected_path}")

    if current_path.stat().st_size <= 1 and expected_path.stat().st_size <= 1:
        return pd.DataFrame(), pd.DataFrame()

    return pd.read_csv(current_path), pd.read_csv(expected_path)


def compare_results(filename: str, atol: float = 1e-8):
    """Compare current notebook output CSV against expected baseline."""
    current, expected = load_csv_pair(filename)
    pdt.assert_frame_equal(current, expected, atol=atol, check_dtype=False)


def test_guide_expected_files_cover_notebook_outputs():
    missing = [name for name in GUIDE_CSV_FILES if not (EXPECTED_DIR / name).exists()]
    assert not missing, f"Missing expected guide CSV baselines: {missing}"


@pytest.mark.parametrize("filename", GUIDE_CSV_FILES)
def test_guide_output_matches_expected_baseline(filename):
    """Compare DataStream guide output CSVs against expected baselines."""
    compare_results(filename)
