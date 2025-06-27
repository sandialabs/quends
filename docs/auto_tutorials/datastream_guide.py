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

# %%
# GX Data Analysis
# ----------------
# Analysis on GX Data

# Specify the file paths
csv_file_path = "gx/tprim_2_0.out.csv"
csv2_file_path = "gx/ensemble/tprim_2_5_a.out.csv"

# Load the data from CSV files
data_stream_csv = qnds.from_csv(csv_file_path)
data_stream_gx = qnds.from_csv(csv2_file_path)

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

# Check if multiple columns are stationary
data_stream_gx.is_stationary(["HeatFlux_st", "Wg_st", "Phi2_t"])

# %%
# Trimming data based to obtain steady-state portion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# %%
# Trim the data based on standard deviation method

# Returns: Dictionary with keys like "results" and "metadata"
trimmed = data_stream_gx.trim(column_name="HeatFlux_st", batch_size=50, method="std")

# Gather results from dictionary
trimmed_df = trimmed["results"]

# Print first 5 rows of dataframe
trimmed_df.head()

# %%
# Trim the data based on rolling variance method
trimmed = data_stream_gx.trim(
    column_name="HeatFlux_st", batch_size=50, method="rolling_variance", threshold=0.10
)

# Gather results
trimmed_df = trimmed["results"]

# Return trimmed data
if trimmed_df is not None:
    print(trimmed_df.head())
else:
    print("Trim returned None (no data trimmed or steady state not found).")

# %%
# Trim the data based on threshold method
trimmed = data_stream_gx.trim(
    column_name="HeatFlux_st", batch_size=50, method="threshold", threshold=0.1
)

# Gather results
trimmed_df = trimmed["results"]

# View trimmed data
trimmed_df.head()

# %%
# Effective Sample Size
# ~~~~~~~~~~~~~~~~~~~~~
#
# Compute Effective Sample Size for specific columns in GX
ess_dict = data_stream_gx.effective_sample_size(column_names=["HeatFlux_st", "Wg_st"])
print(ess_dict)

# %%
# Compute Effective sample size for trimmed data
ess_df = trimmed_df.effective_sample_size()
print(ess_df)

# %%
# UQ Analysis
# -----------
#
# Compute Statistics on trimmed dataframe

stats = trimmed_df.compute_statistics(method="sliding")
print(stats)

stats_df = stats["HeatFlux_st"]

# %%
# Exporter
# Below Displays the information as a DataFrame
exporter = qnds.Exporter()
exporter.display_dataframe(stats_df)

# %%
# Below Displays the information in JSON

exporter.display_json(stats_df)

# %%
# Other statistical methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# Calculate the mean with a window size of 10
mean_df = trimmed_df.mean(window_size=10)
print(mean_df)

# %%
# Calculate the mean with the method of sliding
mean_df = trimmed_df.mean(method="sliding")
print(mean_df)

# %%
# Calculate the mean uncertainty
uq_df = trimmed_df.mean_uncertainty()
print(uq_df)

# %%
# Calculate the mean uncertainty with the method of sliding
uq_df = trimmed_df.mean_uncertainty(method="sliding")
uq_df

# %%
# Calculate the confidence intervale with the trimmed dataframe
ci_df = trimmed_df.confidence_interval()
print(ci_df)

# %%
# Optimal Window
# ~~~~~~~~~~~~~~
#

# %%
# Calulcautes the optimal window size
optimal_df = trimmed_df.optimal_window_size()
print(optimal_df)

# %%
# Cumlative Statistics
cumulative = trimmed_df.cumulative_statistics()
print(cumulative)

cumulative_df = cumulative["HeatFlux_st"]

# %%
# Display Cumulative Statistics as a DataFrame
exporter.display_dataframe(cumulative_df)

# %%
# CGYRO Data Analysis
# ~~~~~~~~~~~~~~~~~~~
#

# %%
# Specify the file paths
csv_file_path = "cgyro/output_nu0_50.csv"
data_stream_cg = qnds.from_csv(csv_file_path)
data_stream_cg.head()

# %%
# Get the number of rows
len(data_stream_cg)

# %%
# Trim the data based on threshold method
trimmed_ = data_stream_cg.trim(column_name="Q_D/Q_GBD", method="std", robust=True)
# View trimmed data
print(trimmed_)


# %%
trimmed_df = trimmed_["results"]
trimmed_df.head()

# %%
# To check if data stream is stationary
data_stream_cg.is_stationary("Q_D/Q_GBD")

# %%
# To Plot for DataStream
plotter = qnds.Plotter()
plotter.trace_plot(data_stream_cg, ["Q_D/Q_GBD"])

# %%
plotter.trace_plot(trimmed_df)

# %%
plotter.steady_state_automatic_plot(data_stream_cg, ["Q_D/Q_GBD"])

# %%
plotter.steady_state_automatic_plot(trimmed_df)

# %%
plotter.steady_state_plot(data_stream_cg, ["Q_D/Q_GBD"], 158.59)

# %%
# To show additional data use:
addition_info = trimmed_df.additional_data(method="sliding")
print(addition_info)

# %%
# To add a reduction factor
addition_info = trimmed_df.additional_data(reduction_factor=0.2)
print(addition_info)
