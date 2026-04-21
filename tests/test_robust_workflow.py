import os
import tempfile
import warnings
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import nbformat
import numpy as np
import pandas as pd
import pandas.testing as pdt
import papermill as pm
import pytest

from quends import DataStream, RobustWorkflow

pytest_plugins = ("tests._shared",)

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
    current, expected = load_csv_pair(filename)
    shared_cols = [c for c in expected.columns if c in current.columns]
    assert shared_cols, f"No shared columns found for {filename}"
    pdt.assert_frame_equal(
        current[shared_cols], expected[shared_cols], atol=atol, check_dtype=False
    )
    print(
        f"Regression test passed for {filename}: current results match expected baseline."
    )


# Execute notebook once before all tests
@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Execute notebook once before running all tests."""
    execute_notebook()


def test_linear_transient_to_plateau():
    compare_results("linear_transient_to_plateau.csv", atol=1e-8)


def test_slope_to_sine_regression():
    compare_results("slope_to_sine_stats.csv", atol=1e-8)


def test_regular_signals_cygro():
    compare_results("regular_signals_cgyro.csv", atol=1e-8)


def test_regular_signals_gx():
    compare_results("regular_signals_gx.csv", atol=1e-8)


def test_non_stat():
    compare_results("non-stat.csv", atol=1e-8)


def test_non_stat_drop():
    compare_results("non-stat-drop.csv", atol=1e-8)


def test_no_sss():
    compare_results("no_sss.csv", atol=1e-8)


def test_batch():
    batch_files = [
        "output_nu0_02_batch.csv",
        "output_nu0_05_batch.csv",
        "output_nu0_10_batch.csv",
        "output_nu0_50_batch.csv",
        "output_nu1_0_batch.csv",
    ]

    # Execute the notebook once
    execute_notebook()

    # Compare each batch result CSV
    for fname in batch_files:
        current, expected = load_csv_pair(fname)
        shared_cols = [c for c in expected.columns if c in current.columns]
        assert shared_cols, f"No shared columns found for {fname}"
        pdt.assert_frame_equal(
            current[shared_cols], expected[shared_cols], atol=1e-8, check_dtype=False
        )
        print(f"Regression test passed for {fname}")


# Fixtures


def make_workflow(operate_safe=True, verbosity=0):
    return RobustWorkflow(
        operate_safe=operate_safe,
        verbosity=verbosity,
        drop_fraction=0.25,
        n_pts_min=10,
        n_pts_frac_min=0.2,
        max_lag_frac=0.5,
        autocorr_sig_level=0.05,
        decor_multiplier=4.0,
        std_dev_frac=0.1,
        fudge_fac=0.1,
        smoothing_window_correction=0.8,
        final_smoothing_window=10,
    )


def make_datastream(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return DataStream(
        pd.DataFrame(
            {
                "time": np.arange(n, dtype=float),
                "A": rng.normal(5.0, 0.3, n),
            }
        )
    )


def make_nonstationary_datastream(n=200):
    return DataStream(
        pd.DataFrame(
            {
                "time": np.arange(n, dtype=float),
                "A": np.linspace(0, 100, n),
            }
        )
    )


# process_irregular_stream: operate_safe=True


class TestProcessIrregularStreamSafe:

    def test_returns_nan_mean(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert np.isnan(result["A"]["mean"])

    def test_returns_nan_uncertainty(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert np.isnan(result["A"]["mean_uncertainty"])

    def test_returns_nan_confidence_interval(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        ci = result["A"]["confidence_interval"]
        assert np.isnan(ci[0]) and np.isnan(ci[1])

    def test_returns_nan_sss_start(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert np.isnan(result["A"]["sss_start"])

    def test_status_is_no_stat_steady_state(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert result["A"]["metadata"]["status"] == "NoStatSteadyState"

    def test_mitigation_is_drop(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert result["A"]["metadata"]["mitigation"] == "Drop"

    def test_start_time_recorded(self):
        wf = make_workflow(operate_safe=True)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A", start_time=10.0)
        assert result["A"]["start_time"] == 10.0


# process_irregular_stream


class TestProcessIrregularStreamAdHoc:

    def test_returns_finite_mean(self):
        wf = make_workflow(operate_safe=False)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert np.isfinite(result["A"]["mean"])

    def test_uncertainty_equals_mean(self):
        """Ad-hoc sets uncertainty = mean (100% uncertainty)."""
        wf = make_workflow(operate_safe=False)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert result["A"]["mean_uncertainty"] == result["A"]["mean"]

    def test_confidence_interval_is_symmetric_around_mean(self):
        wf = make_workflow(operate_safe=False)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        mean = result["A"]["mean"]
        ci = result["A"]["confidence_interval"]
        assert ci[0] == pytest.approx(mean - mean)
        assert ci[1] == pytest.approx(mean + mean)

    def test_sss_start_is_at_two_thirds_mark(self):
        """sss_start should correspond to the 2/3 index of the data."""
        wf = make_workflow(operate_safe=False)
        n = 200
        ds = make_datastream(n=n)
        result = wf.process_irregular_stream(ds, "A", start_time=0.0)
        n_66pc = (n * 2) // 3
        expected_time = ds.data.iloc[n_66pc]["time"]
        assert result["A"]["sss_start"] == pytest.approx(expected_time)

    def test_status_is_no_stat_steady_state(self):
        wf = make_workflow(operate_safe=False)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert result["A"]["metadata"]["status"] == "NoStatSteadyState"

    def test_mitigation_is_adhoc(self):
        wf = make_workflow(operate_safe=False)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A")
        assert result["A"]["metadata"]["mitigation"] == "AdHoc"

    def test_respects_start_time_filter(self):
        """Data before start_time should be excluded from ad-hoc mean."""
        wf = make_workflow(operate_safe=False)
        ds = make_datastream(n=200)
        result_t0 = wf.process_irregular_stream(ds, "A", start_time=0.0)
        result_t100 = wf.process_irregular_stream(ds, "A", start_time=100.0)
        # means will differ because different data slices are used
        assert result_t0["A"]["mean"] != result_t100["A"]["mean"]

    def test_start_time_recorded(self):
        wf = make_workflow(operate_safe=False)
        ds = make_datastream()
        result = wf.process_irregular_stream(ds, "A", start_time=5.0)
        assert result["A"]["start_time"] == 5.0

    def test_nan_values_dropped_before_mean(self):
        """NaNs in the column should not propagate to the mean."""
        wf = make_workflow(operate_safe=False)
        df = pd.DataFrame(
            {
                "time": np.arange(200, dtype=float),
                "A": np.where(np.arange(200) % 10 == 0, np.nan, 5.0),
            }
        )
        ds = DataStream(df)
        result = wf.process_irregular_stream(ds, "A")
        assert np.isfinite(result["A"]["mean"])


# process_data_steam: verbosity print paths


class TestProcessDataSteamVerbosity:

    def test_verbosity_1_prints_size_info(self, capsys):
        wf = make_workflow(verbosity=1)
        ds = make_nonstationary_datastream()
        wf.process_data_steam(ds, "A")
        captured = capsys.readouterr()
        assert "Original size" in captured.out
        assert "start time" in captured.out

    def test_verbosity_0_no_size_output(self, capsys):
        wf = make_workflow(verbosity=0)
        ds = make_nonstationary_datastream()
        wf.process_data_steam(ds, "A")
        captured = capsys.readouterr()
        assert "Original size" not in captured.out

    def test_verbosity_1_prints_not_stationary(self, capsys):
        wf = make_workflow(verbosity=1)
        ds = make_nonstationary_datastream()
        wf.process_data_steam(ds, "A")
        captured = capsys.readouterr()
        assert "not stationary" in captured.out.lower()

    def test_verbosity_1_prints_no_sss_when_trimming_fails(
        self, capsys, slope_to_stationary_df
    ):
        """
        When stationary=True but no SSS is found after trimming,
        verbosity=1 should print the no-SSS message.
        """
        wf = make_workflow(verbosity=1, operate_safe=False)
        ds = DataStream(slope_to_stationary_df)

        with patch(
            "quends.TrimDataStreamOperation.__call__",
            return_value=MagicMock(
                __len__=lambda s: 0, data=pd.DataFrame(columns=["time", "A"])
            ),
        ):
            wf.process_data_steam(ds, "A")

        captured = capsys.readouterr()
        assert "No statistical steady state" in captured.out


# process_data_steam


class TestProcessDataSteamNoSSSAfterTrim:

    def test_falls_back_to_irregular_when_trim_empty(self, slope_to_stationary_df):
        """
        If trimmed_stream has len <= 1, process_irregular_stream should be called.
        With operate_safe=True this means NaN mean.
        """
        wf = make_workflow(operate_safe=True)
        ds = DataStream(slope_to_stationary_df)

        empty_stream = MagicMock()
        empty_stream.__len__ = lambda s: 0
        empty_stream.data = pd.DataFrame(columns=["time", "A"])

        with patch(
            "quends.TrimDataStreamOperation.__call__", return_value=empty_stream
        ):
            result = wf.process_data_steam(ds, "A")

        assert np.isnan(result["A"]["mean"])
        assert result["A"]["metadata"]["mitigation"] == "Drop"

    def test_falls_back_to_adhoc_when_trim_empty_unsafe(self, slope_to_stationary_df):
        wf = make_workflow(operate_safe=False)
        ds = DataStream(slope_to_stationary_df)

        empty_stream = MagicMock()
        empty_stream.__len__ = lambda s: 0
        empty_stream.data = pd.DataFrame(columns=["time", "A"])

        with patch(
            "quends.TrimDataStreamOperation.__call__", return_value=empty_stream
        ):
            result = wf.process_data_steam(ds, "A")

        assert result["A"]["metadata"]["mitigation"] == "AdHoc"


# plot_signal_basic_stats


class TestPlotSignalBasicStats:

    def _make_stats(self, col="A", mean=5.0, start_time=0.0, sss_start=10.0):
        return {
            col: {
                "mean": mean,
                "mean_uncertainty": 0.5,
                "confidence_interval": (mean - 0.5, mean + 0.5),
                "sss_start": sss_start,
                "start_time": start_time,
            }
        }

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_runs_without_error_no_stats(self, mock_close, mock_show):
        wf = make_workflow()
        ds = make_datastream()
        wf.plot_signal_basic_stats(ds, "A")
        mock_show.assert_called_once()
        assert mock_close.call_count >= 1

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_runs_without_error_with_stats(self, mock_close, mock_show):
        wf = make_workflow()
        ds = make_datastream()
        stats = self._make_stats(start_time=0.0)
        wf.plot_signal_basic_stats(ds, "A", stats=stats)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_start_time_axvline_drawn_when_positive(self, mock_close, mock_show):
        """start_time > 0 should trigger an extra axvline for the start time."""
        wf = make_workflow()
        ds = make_datastream()
        stats = self._make_stats(start_time=20.0)

        axvline_calls = []
        with patch(
            "matplotlib.axes.Axes.axvline",
            side_effect=lambda *a, **kw: axvline_calls.append(kw.get("x")),
        ):
            wf.plot_signal_basic_stats(ds, "A", stats=stats)

        # Should have both start_time and sss_start vlines
        assert 20.0 in axvline_calls

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_start_time_axvline_not_drawn_when_zero(self, mock_close, mock_show):
        """start_time == 0 should NOT draw the start_time vline."""
        wf = make_workflow()
        ds = make_datastream()
        stats = self._make_stats(start_time=0.0)

        axvline_calls = []
        with patch(
            "matplotlib.axes.Axes.axvline",
            side_effect=lambda *a, **kw: axvline_calls.append(kw.get("x")),
        ):
            wf.plot_signal_basic_stats(ds, "A", stats=stats)

        assert 0.0 not in axvline_calls

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_label_applied_when_provided(self, mock_close, mock_show):
        """Providing a label should set the axes title."""
        wf = make_workflow()
        ds = make_datastream()

        set_title_calls = []
        with patch(
            "matplotlib.axes.Axes.set_title",
            side_effect=lambda *a, **kw: set_title_calls.append(a[0]),
        ):
            wf.plot_signal_basic_stats(ds, "A", label="My Signal")

        assert "My Signal" in set_title_calls

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_no_title_when_label_is_none(self, mock_close, mock_show):
        """No label = no set_title call."""
        wf = make_workflow()
        ds = make_datastream()

        set_title_calls = []
        with patch(
            "matplotlib.axes.Axes.set_title",
            side_effect=lambda *a, **kw: set_title_calls.append(a[0]),
        ):
            wf.plot_signal_basic_stats(ds, "A", label=None)

        assert set_title_calls == []

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_three_stat_lines_drawn_with_stats(self, mock_close, mock_show):
        """Mean, upper CI, and lower CI lines should each produce an ax.plot call."""
        wf = make_workflow()
        ds = make_datastream()
        stats = self._make_stats()

        plot_labels = []
        original_plot = __import__("matplotlib").axes.Axes.plot

        def capture_plot(self_ax, *args, **kwargs):
            plot_labels.append(kwargs.get("label"))
            return original_plot(self_ax, *args, **kwargs)

        with patch("matplotlib.axes.Axes.plot", capture_plot):
            wf.plot_signal_basic_stats(ds, "A", stats=stats)

        assert "Mean" in plot_labels
        assert "95% Conf. Int." in plot_labels
