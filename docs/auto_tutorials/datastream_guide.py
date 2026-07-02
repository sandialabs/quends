r"""
DataStream Class
----------------
This tutorial demonstrates the usage of the DataStream class,
which provides methods for analyzing time-series data.

The following features are:
    - **Trimming**: Identifies steady-state regions in data.
    - **Statistical Analysis**: Computes mean, standard deviation, confidence intervals, and cumulative statistics.
    - **Stationarity Testing**: Uses the Augmented Dickey-Fuller test.
    - **Effective Sample Size (ESS)**: Estimates the independent sample size.
    - **Optimal Window Size**: Determines the best window for data smoothing.
"""

# %%
# Import DataStream
import quends as qnds
from quends.base.trim import (
    NoiseThresholdTrimStrategy,
    QuantileTrimStrategy,
    RollingVarianceThresholdTrimStrategy,
    TrimDataStreamOperation,
)

# %%
# GX Data Analysis
# ----------------
# Analysis on GX Data

# Specify the file paths
csv_file_path = "gx/tprim_2_0.out.csv"
csv2_file_path = "gx/ensemble/tprim_2_5_a.out.csv"

# Load the data from CSV files
data_stream_csv = qnds.from_csv(csv_file_path, "HeatFlux_st")
data_stream_gx = qnds.from_csv(csv2_file_path, "HeatFlux_st")

# Display the first few rows of the GX data
data_stream_gx.head()

# %%
# Get available variables
data_stream_gx.variables()

# %%
# Get number of rows from the following data in GX
len(data_stream_gx)

# %%
# Stationary Check
# ~~~~~~~~~~~~~~~~
#

# Check if a single column is stationary
data_stream_gx.is_stationary("HeatFlux_st")

# Check stationarity for several variables. With the single-variable API,
# each column is loaded into its own DataStream.
for _var in ["HeatFlux_st", "Wg_st", "Phi2_t"]:
    print(_var, qnds.from_csv(csv2_file_path, _var).is_stationary(_var))

# %%
# Trimming data based to obtain steady-state portion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# %%
# Trim the data based on standard deviation method (Quantile strategy)
# Use the strategy-operation pattern from quends.base.trim directly.

strategy = QuantileTrimStrategy(window_size=50, robust=True)
op = TrimDataStreamOperation(strategy=strategy)
trimmed = op(data_stream_gx, column_name="HeatFlux_st")

# Print first 5 rows of dataframe
trimmed.head()

# %%
# Trim the data based on rolling variance method
strategy = RollingVarianceThresholdTrimStrategy(window_size=50, threshold=0.10)
op = TrimDataStreamOperation(strategy=strategy)
trimmed = op(data_stream_gx, column_name="HeatFlux_st")

# Gather results
trimmed.head()

# %%
# Trim the data based on noise threshold method
strategy = NoiseThresholdTrimStrategy(window_size=50, threshold=0.1)
op = TrimDataStreamOperation(strategy=strategy)
trimmed = op(data_stream_gx, column_name="HeatFlux_st")

# View trimmed data
trimmed.head()

# %%
# Effective Sample Size
# ~~~~~~~~~~~~~~~~~~~~~
#
# Compute Effective Sample Size for specific columns in GX. With the
# single-variable API, each column is loaded into its own DataStream.
for _var in ["HeatFlux_st", "Wg_st"]:
    print(_var, qnds.from_csv(csv2_file_path, _var).effective_sample_size())

# %%
# Compute Effective sample size for trimmed data
ess_df = trimmed.effective_sample_size()
print(ess_df)

# %%
# UQ Analysis
# -----------
#
# Compute Statistics on trimmed dataframe

stats = trimmed.compute_statistics(method="sliding")
print(stats)

stats_df = stats["HeatFlux_st"]

# %%
# Exporter
# Below Displays the information as a DataFrame
exporter = qnds.Exporter()
exporter.display_dataframe(stats)

# %%
# Below Displays the information in JSON

exporter.display_json(stats)

# %%
# Other statistical methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# Calculate the mean with a window size of 10
stats = trimmed.compute_statistics(window_size=10)
mean_df = {col: values["mean"] for col, values in stats.items() if "mean" in values}
print(mean_df)

# %%
# Calculate the mean with the method of sliding
stats = trimmed.compute_statistics(method="sliding")
mean_df = {col: values["mean"] for col, values in stats.items() if "mean" in values}
print(mean_df)

# %%
# Calculate the mean uncertainty
stats = trimmed.compute_statistics()
uq_df = {
    col: values["mean_uncertainty"]
    for col, values in stats.items()
    if "mean_uncertainty" in values
}
print(uq_df)

# %%
# Calculate the mean uncertainty with the method of sliding
stats = trimmed.compute_statistics(method="sliding")
uq_df = {
    col: values["mean_uncertainty"]
    for col, values in stats.items()
    if "mean_uncertainty" in values
}
uq_df

# %%
# Calculate the confidence intervale with the trimmed dataframe
stats = trimmed.compute_statistics()
ci_df = {
    col: values["confidence_interval"]
    for col, values in stats.items()
    if "confidence_interval" in values
}
print(ci_df)


# %%
# Cumlative Statistics
cumulative = trimmed.cumulative_statistics()
print(cumulative)

cumulative_df = cumulative["HeatFlux_st"]

# %%
# Display Cumulative Statistics as a DataFrame
exporter.display_dataframe(cumulative)

# %%
# CGYRO Data Analysis
# ~~~~~~~~~~~~~~~~~~~
#

# %%
# Specify the file paths
csv_file_path = "cgyro/output_nu0_50.csv"
data_stream_cg = qnds.from_csv(csv_file_path, "Q_D/Q_GBD")
data_stream_cg.head()

# %%
# Get the number of rows
len(data_stream_cg)

# %%
# Trim the CGYRO data using the Quantile (std) strategy
strategy = QuantileTrimStrategy(robust=True)
op = TrimDataStreamOperation(strategy=strategy)
trimmed_ = op(data_stream_cg, column_name="Q_D/Q_GBD")
# View trimmed data
print(trimmed_)


# %%
trimmed_.head()

# %%
# To check if data stream is stationary
data_stream_cg.is_stationary("Q_D/Q_GBD")

# %%
# To Plot for DataStream
plotter = qnds.Plotter()
plot = plotter.trace_plot(data_stream_cg, ["Q_D/Q_GBD"])

# %%
plot = plotter.steady_state_automatic_plot(
    data_stream_cg, variables_to_plot=["Q_D/Q_GBD"]
)

# %%
plot = plotter.steady_state_plot(data_stream_cg, variables_to_plot=["Q_D/Q_GBD"])

# %%
# To show additional data use:
addition_info = trimmed.additional_data(method="sliding")
print(addition_info)

# %%
# To add a reduction factor
addition_info = trimmed.additional_data(reduction_factor=0.2)
print(addition_info)
