import numpy as np
import pandas as pd
import pytest
from quends import DataStream, RobustWorkflow


def test_process_data_stream_regular_case():
    """Test that lines 295-306 in robust_workflow.py work correctly for regular case."""
    # Create a workflow instance
    wf = RobustWorkflow(
        operate_safe=True,
        verbosity=0,
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
    
    # Create a stationary data stream with clear SSS
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "time": np.arange(n, dtype=float),
        "A": rng.normal(5.0, 0.3, n),
    })
    ds = DataStream(df)
    
    # Process the data stream
    result = wf.process_data_stream(ds, "A", start_time=0.0)
    
    # Verify the results contain expected fields from lines 295-306
    assert "A" in result
    assert "sss_start" in result["A"]
    assert "metadata" in result["A"]
    assert "status" in result["A"]["metadata"]
    assert "mitigation" in result["A"]["metadata"]
    assert "start_time" in result["A"]
    
    # Verify specific values from lines 295-306
    assert result["A"]["metadata"]["status"] == "Regular"
    assert result["A"]["metadata"]["mitigation"] == "None"
    assert result["A"]["start_time"] == 0.0
    
    # Verify sss_start is set (should be the first time value after trimming)
    assert isinstance(result["A"]["sss_start"], (int, float))
    assert result["A"]["sss_start"] >= 0.0
    
    # Verify statistics are computed
    assert "mean" in result["A"]
    assert "mean_uncertainty" in result["A"]
    assert "confidence_interval" in result["A"]
    
    print("Test passed: Lines 295-306 work correctly for regular case")


if __name__ == "__main__":
    test_process_data_stream_regular_case()