r"""
Ensemble Analysis
=================
This tutorial analyses **multiple simulation runs together** with the
:class:`~quends.Ensemble` object, using the GX members in
``examples/data/gx/ensemble``.

Instead of one long trace, an ensemble combines several shorter runs. QUENDS
provides three ways to turn an ensemble into a mean with an honest uncertainty:

* **Ensemble Average** -- average the members onto a common grid, then analyse
  that single averaged trace,
* **Serialization (pooled block means)** -- pool the per-member block means,
* **Inverse-Variance-Weighted (IVW)** -- weight each member's mean by its
  inverse variance.

For single-trace analysis, see the *DataStream Class* guide. Trim/statistics
parameters follow the QUENDS paper analysis notebooks for the stellarator GX
ensemble (``method="threshold"``, ``window_size=50``, ``start_time=100``,
``threshold=0.19``).
"""

# %%
# Import QUENDS
import glob
from pathlib import Path

import quends as qnds


def example_data_dir() -> Path:
    """Find the shared example data directory during script or gallery runs."""
    starts = []
    if "__file__" in globals():
        starts.append(Path(__file__).resolve())
    starts.append(Path.cwd().resolve())

    for start in starts:
        for parent in [start, *start.parents]:
            for candidate in (parent / "examples" / "data", parent / "data"):
                if candidate.is_dir():
                    return candidate
    raise FileNotFoundError("Could not locate examples/data")


COL = "HeatFlux_st"
DATA_DIR = example_data_dir()
plotter = qnds.Plotter()

# %%
# Data Loading
# ------------
# ``Ensemble.from_files`` loads each CSV into a member ``DataStream``.
gx_files = sorted(glob.glob(str(DATA_DIR / "gx" / "ensemble" / "tprim_2_5_*.out.csv")))
ens = qnds.Ensemble.from_files(gx_files, COL)
print("ensemble members:", len(ens.members()))

# %%
# Plotting all members together with the ensemble average.
plot = plotter.plot_ensemble_with_average(
    ens, variables_to_plot=[COL], condensed_legend=True, show=True
)

# %%
# Ensemble Average Approach
# -------------------------
# Average the members onto a common time grid, then analyse that single
# averaged trace exactly like a single run.

# %%
# Build the averaged DataStream and check stationarity.
avg = ens.compute_average_ensemble()
print("averaged trace rows:", len(avg))
print("avg is_stationary:", avg.is_stationary(COL))

# %%
# Trim the averaged trace, then read off its ESS.
avg_trimmed = avg.trim(
    method="threshold", window_size=50, start_time=100, threshold=0.19
)
print("avg sss_start:", avg_trimmed.trim_metadata.get("sss_start"))
print("avg ESS:", avg_trimmed.effective_sample_size())

# %%
# The Ensemble Average estimate of the mean and its uncertainty. We analyse the
# **trimmed** averaged trace (average -> trim -> statistics), so the transient is
# removed before the mean and its standard error are computed. NB: the
# ``ensemble_average`` estimator does **not** trim internally -- running it on a
# raw, un-trimmed average folds the transient into the variance and greatly
# inflates the uncertainty.
ea_stats = avg_trimmed.compute_statistics(method="non-overlapping")[COL]
print("Ensemble Average ->", ea_stats)

# %%
# Serialization (Pooled Block Means) Approach
# -------------------------------------------
# Rather than averaging the traces, pool the per-member block means. Trim every
# member first, then aggregate across members.
ens_trimmed = ens.trim(
    column_name=COL, method="threshold", window_size=50, start_time=100, threshold=0.19
)
print("members stationary:", ens.is_stationary(COL)["results"])
print("ESS (pooled_block_means):", ens_trimmed.effective_sample_size(COL)["results"])

# %%
# The serialization estimate of the mean and its uncertainty, on the trimmed
# members.
ser_stats = ens_trimmed.compute_uncertainty(
    method="pooled_block_means", column_name=COL
)["results"][COL]
print("Serialization ->", ser_stats)

# %%
# Inverse-Variance-Weighted (IVW) Approach
# ----------------------------------------
# Combine the per-member means weighting each by its inverse variance, so
# better-resolved members count more. Trimming/stationarity are as above.
print("ESS (ivw):", ens_trimmed.effective_sample_size(COL, technique="ivw")["results"])
ivw_stats = ens_trimmed.compute_uncertainty(method="ivw", column_name=COL)["results"][
    COL
]
print("IVW ->", ivw_stats)

# %%
# Computed on the trimmed steady-state data, the three approaches give
# consistent means with comparable uncertainties -- a useful cross-check on an
# ensemble.
for name, r in [
    ("Ensemble Average", ea_stats),
    ("Serialization", ser_stats),
    ("IVW", ivw_stats),
]:
    print(
        f"{name:18s} mean={r.get('mean'):.4f}  uncertainty={r.get('uncertainty', r.get('mean_uncertainty')):.4f}"
    )
