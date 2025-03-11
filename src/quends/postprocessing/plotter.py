import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from quends.base.data_stream import DataStream  # Adjust the import if necessary
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from quends.base.ensemble import Ensemble

class Plotter:
    """
    A class that encapsulates plotting functionality for time series data.
    """

    def __init__(self, output_dir="results_figures"):
        """
        Initialize the Plotter.

        Args:
            output_dir (str): Directory to save the generated plots.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def format_dataset_name(dataset_name):
        """
        Format the dataset name for display and file naming.

        Args:
            dataset_name (str): The original dataset name.
        
        Returns:
            str: A formatted dataset name.
        """
        return dataset_name.replace("_", " ").title()

    def _prepare_data_frames(self, data):
        """
        Prepare a dictionary of DataFrames from the input data.

        Args:
            data (DataStream or dict): A DataStream instance or a dictionary of DataFrames.
        
        Returns:
            dict: A dictionary of DataFrames keyed by dataset name.
        """
        if isinstance(data, DataStream):
            return {"DataStream": data.df}
        elif isinstance(data, Ensemble):
            return {f"DataStream {k}": data.data_streams[k].df for k in range(len(data))}
        elif isinstance(data, dict):
            return data
        else:
            raise ValueError("Input data must be a DataStream instance or a dictionary of DataFrames")

    
    def _calc_fig_size(self, num_cols, num_rows):
        """
        Calculate an automatic figure size based on the number of subplot columns and rows.
        
        Args:
            num_cols (int): Number of subplot columns.
            num_rows (int): Number of subplot rows.
        
        Returns:
            tuple: (width, height) for the figure.
        """
        width = max(8, num_cols * 3)
        height = max(6, num_rows * 3)
        return (width, height)
    
    def trace_plot(self, data, variables_to_plot=None, save=False):
        """
        Plot individual (trace) time series data from a DataStream or a dictionary of DataFrames.
        The resulting plots are displayed and optionally saved if 'save' is True.

        Args:
            data (DataStream or dict): A DataStream instance or dictionary of DataFrames.
            variables_to_plot (list, optional): List of variables to plot. If None,
                all columns (except 'time') from the first DataFrame are used.
            save (bool, optional): If True, save the generated plots to the output directory.
                                   Defaults to False.
        """
        data_frames = self._prepare_data_frames(data)

        # If no variables provided, infer from the first DataFrame (excluding 'time')
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [col for col in first_df.columns if col != "time"]
        else:
            variables_to_plot = [var for var in variables_to_plot if var != "time"]

        # Create a trace plot for each dataset in the dictionary
        for dataset_name, df in data_frames.items():
            time_series = df["time"]
            num_traces = len(variables_to_plot)
            num_cols = min(5, num_traces)
            num_rows = (num_traces + num_cols - 1) // num_cols  # Ceiling division
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            #fig_width = num_cols * 4
            #fig_height = num_rows * 4

            #fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
            # Ensure axes is always a flat list
            if num_traces == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for j, column in enumerate(variables_to_plot):
                axes[j].plot(time_series, df[column], label=column)
                axes[j].set_xlabel("Time")
                axes[j].set_title(column)
                axes[j].legend(fontsize='small')
                axes[j].grid(True)

            # Remove any unused subplots
            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(f"Time Series Plots for {self.format_dataset_name(dataset_name)}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(self.output_dir, f"time_series_{self.format_dataset_name(dataset_name)}.png")
                plt.savefig(save_path)
            plt.show()
            plt.close()

        return axes
    
    def trace_plot_with_mean(self, data, variables_to_plot=None, save=False):
        """
        Plot individual (trace) time series data from a DataStream or a dictionary of DataFrames.
        The resulting plots are displayed and optionally saved if 'save' is True.

        Args:
            data (DataStream or dict): A DataStream instance or dictionary of DataFrames.
            variables_to_plot (list, optional): List of variables to plot. If None,
                all columns (except 'time') from the first DataFrame are used.
            save (bool, optional): If True, save the generated plots to the output directory.
                                   Defaults to False.
        """
        data_frames = self._prepare_data_frames(data)

        # If no variables provided, infer from the first DataFrame (excluding 'time')
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [col for col in first_df.columns if col != "time"]
        else:
            variables_to_plot = [var for var in variables_to_plot if var != "time"]

        # Create a trace plot for each dataset in the dictionary
        for dataset_name, df in data_frames.items():
            time_series = df["time"]
            num_traces = len(variables_to_plot)
            num_cols = min(5, num_traces)
            num_rows = (num_traces + num_cols - 1) // num_cols  # Ceiling division
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            #fig_width = num_cols * 4
            #fig_height = num_rows * 4

            #fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
            # Ensure axes is always a flat list
            if num_traces == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for j, column in enumerate(variables_to_plot):
                ds = DataStream(df)
                tds = ds.trim(column)
                m = tds.mean()[column]["mean"]
                l, u = tds.confidence_interval()[column]["confidence interval"]
                axes[j].plot(time_series, df[column], label=column)
                axes[j].plot(tds.df["time"], m * np.ones(len(tds.df["time"])), color="r", label="mean")
                axes[j].fill_between(tds.df["time"], l, u, color="r", alpha=0.2, label="mean uncertainty")
                axes[j].set_xlabel("Time")
                axes[j].set_title(column)
                axes[j].legend(fontsize='small')
                axes[j].grid(True)

            # Remove any unused subplots
            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(f"Time Series Plots for {self.format_dataset_name(dataset_name)}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(self.output_dir, f"time_series_{self.format_dataset_name(dataset_name)}.png")
                plt.savefig(save_path)
            plt.show()
            plt.close()
        

    def ensemble_trace_plot(self, data, variables_to_plot=None, save=False):
        """
        Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.
        The resulting plots are displayed and optionally saved if 'save' is True.

        Args:
            data (DataStream or dict): A DataStream instance or dictionary of DataFrames representing ensemble members.
            variables_to_plot (list, optional): List of variables to plot. If None,
                all columns (except 'time') from the first DataFrame are used.
            save (bool, optional): If True, save the generated plots to the output directory.
                                   Defaults to False.
        """
        data_frames = self._prepare_data_frames(data)

        # If no variables provided, infer from the first DataFrame (excluding 'time')
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [col for col in first_df.columns if col != "time"]
        else:
            variables_to_plot = [var for var in variables_to_plot if var != "time"]

        for var in variables_to_plot:
            plt.figure(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                plt.plot(df["time"], df[var], label=dataset_name)
            plt.xlabel("Time")
            plt.title(f"Ensemble Time Series for {var}")
            plt.legend()
            plt.tight_layout()
            if save:
                save_path = os.path.join(self.output_dir, f"ensemble_time_series_{var}.png")
                plt.savefig(save_path)
            plt.show()
            plt.close()

    def ensemble_trace_plot_with_mean(self, data, variables_to_plot=None, save=False):
        """
        Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.
        The resulting plots are displayed and optionally saved if 'save' is True.

        Args:
            data (DataStream or dict): A DataStream instance or dictionary of DataFrames representing ensemble members.
            variables_to_plot (list, optional): List of variables to plot. If None,
                all columns (except 'time') from the first DataFrame are used.
            save (bool, optional): If True, save the generated plots to the output directory.
                                   Defaults to False.
        """
        data_frames = self._prepare_data_frames(data)

        # If no variables provided, infer from the first DataFrame (excluding 'time')
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [col for col in first_df.columns if col != "time"]
        else:
            variables_to_plot = [var for var in variables_to_plot if var != "time"]

        for var in variables_to_plot:
            plt.figure(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                plt.plot(df["time"], df[var], label=dataset_name)
            plt.xlabel("Time")
            plt.title(f"Ensemble Time Series for {var}")
            plt.legend()
            plt.tight_layout()
            if save:
                save_path = os.path.join(self.output_dir, f"ensemble_time_series_{var}.png")
                plt.savefig(save_path)
            plt.show()
            plt.close()

    def steady_state_automatic_plot(self, data, variables_to_plot=None, window_size=10, start_time=0.0,
                          method="std", threshold=None, robust=True, save=False):
        """
        Plot steady state detection for each variable in the data. For each variable, the method uses the
        DataStream.trim() function to estimate the steady state start time. If a steady state is detected,
        the function plots the original time series along with:
        
            - A vertical dashed red line indicating the steady state start time.
            - A horizontal green line at the overall mean (computed from data after the steady state start).
            - Shaded regions representing ±1, ±2, and ±3 standard deviations.

        If no steady state is detected for a variable, the full signal is plotted and a message is printed.

        Args:
            data (DataStream or dict): A DataStream instance or a dictionary of DataFrames.
            variables_to_plot (list, optional): List of variables to plot. If None,
                all columns (except 'time') from the first DataFrame are used.
            window_size (int, optional): Window size to use in the trim() function.
            start_time (float, optional): Start time for steady state detection.
            method (str, optional): Method to use for steady state detection ('std', 'threshold', or 'rolling_variance').
            threshold (float, optional): Threshold value required for 'threshold' or 'rolling_variance' methods.
            robust (bool, optional): Whether to use robust statistics (median/MAD) in the 'std' method.
            save (bool, optional): If True, save the plot to disk. Defaults to False.
        """
        data_frames = self._prepare_data_frames(data)

        # If no variables provided, infer from the first DataFrame (excluding 'time')
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [col for col in first_df.columns if col != "time"]
        else:
            variables_to_plot = [var for var in variables_to_plot if var != "time"]

        for dataset_name, df in data_frames.items():
            # Create one subplot per variable (stacked vertically)
            time_series = df["time"]
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            if num_vars == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]
                # Create a DataStream from the DataFrame and try to trim using the specified variable
                ds = DataStream(df)
                trimmed_ds = ds.trim(column, window_size=window_size, start_time=start_time,
                                      method=method, threshold=threshold, robust=robust)
                if trimmed_ds is not None:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()
                    
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(x=steady_state_start, color='r', linestyle='--', label='Steady State Start')
                    ax.axhline(y=overall_mean, color='g', linestyle='-', label='Mean')
                    ax.fill_between(time[time >= steady_state_start],
                                    overall_mean - overall_std,
                                    overall_mean + overall_std,
                                    color='blue', alpha=0.3, label='1 Std Dev')
                    ax.fill_between(time[time >= steady_state_start],
                                    overall_mean - 2 * overall_std,
                                    overall_mean + 2 * overall_std,
                                    color='yellow', alpha=0.2, label='2 Std Dev')
                    ax.fill_between(time[time >= steady_state_start],
                                    overall_mean - 3 * overall_std,
                                    overall_mean + 3 * overall_std,
                                    color='red', alpha=0.1, label='3 Std Dev')
                    ax.set_title(column)
                    ax.set_xlabel('Time')
                    ax.set_ylabel(column)
                    ax.legend(fontsize='small')
                    ax.grid(True)
                else:
                    # If trim did not detect a steady state, plot the full signal and print a message.
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.set_title(column)
                    ax.set_xlabel('Time')
                    ax.set_ylabel(column)
                    ax.legend(fontsize='small')
                    ax.grid(True)
                    print(f"{column} is stationary but steady state not achieved. Run longer.")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(self.output_dir, f"steady_state_detection_{self.format_dataset_name(dataset_name)}.png")
                plt.savefig(save_path)
            plt.show()
            plt.close(fig)


    def steady_state_plot(self, data, variables_to_plot=None, steady_state_start=None, save= False):
        """
        Plot steady state detection for each variable in the data using a user-supplied steady state start.
        The user can provide a single float (applied to all variables) or a dictionary mapping variable names to floats.
        For each variable, if a steady state start is provided, the plot displays:
            - The full signal.
            - A vertical dashed red line at the given steady state start.
            - A horizontal green line for the mean (after steady state).
            - Shaded regions for ±1, ±2, and ±3 standard deviations (after steady state).
        If no steady state start is provided for a variable, only the raw signal is plotted and a message is printed.

        Args:
            data (DataStream or dict): A DataStream instance or dictionary of DataFrames.
            variables_to_plot (list, optional): List of variables to plot. If None,
                all columns (except 'time') from the first DataFrame are used.
            steady_state_start (float or dict, optional): Either a single steady state start (float) 
                applied to all variables or a dictionary mapping variable names to steady state start values.
            save (bool): If True, the generated plots are saved to the output directory.
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [col for col in first_df.columns if col != "time"]
        else:
            variables_to_plot = [var for var in variables_to_plot if var != "time"]

        for dataset_name, df in data_frames.items():
            if "time" not in df.columns:
                raise ValueError(f"DataFrame for '{dataset_name}' is missing a 'time' column.")
            
            time_series = df["time"]
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            if num_vars == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for j, column in enumerate(variables_to_plot):
                ax = axes[j]
                # Determine the manual steady state start for this column.
                if isinstance(steady_state_start, dict):
                    manual_ss = steady_state_start.get(column, None)
                else:
                    manual_ss = steady_state_start  # Could be a float or None.
                time = df["time"]
                signal = df[column]
                
                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()
                    
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(x=manual_ss, color='r', linestyle='--', label='Steady State Start')
                    ax.axhline(y=overall_mean, color='g', linestyle='-', label='Mean')
                    ax.fill_between(time[time >= manual_ss],
                                    overall_mean - overall_std,
                                    overall_mean + overall_std,
                                    color='blue', alpha=0.3, label='1 Std Dev')
                    ax.fill_between(time[time >= manual_ss],
                                    overall_mean - 2 * overall_std,
                                    overall_mean + 2 * overall_std,
                                    color='yellow', alpha=0.2, label='2 Std Dev')
                    ax.fill_between(time[time >= manual_ss],
                                    overall_mean - 3 * overall_std,
                                    overall_mean + 3 * overall_std,
                                    color='red', alpha=0.1, label='3 Std Dev')
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(f"For {column}, no manual steady state start provided. Plotting raw signal.")
                
                ax.set_title(column)
                ax.set_xlabel('Time')
                ax.set_ylabel(column)
                ax.legend(fontsize='small')
                ax.grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(self.output_dir, f"steady_state_manual_{self.format_dataset_name(dataset_name)}.png")
                plt.savefig(save_path)
            plt.show()
            plt.close(fig)

    
    def plot_acf(self, data, alpha=0.05, column=None, ax=None):
        """
        Plot the Autocorrelation Function (ACF) for a given data stream or array-like object.
        
        If 'data' is a DataStream, the function extracts the specified column (or the first column 
        that is not "time" if not specified). Otherwise, data is assumed to be 1D array-like.
        
        The function computes:
          - nlags = int(n / 3), where n is the number of observations.
          - ACF values using statsmodels.tsa.stattools.acf.
          - A 95% confidence interval as: conf_interval = z_critical / sqrt(n),
            where z_critical is computed from the two-tailed test.
        
        If an axis (ax) is provided, the plot is drawn on that axis; otherwise, a new figure is created.
        
        Args:
            data (DataStream or array-like): The data to plot.
            alpha (float): Significance level for the confidence interval (default: 0.05).
            column (str, optional): Column name to use if data is a DataStream. Defaults to the first non-'time' column.
            ax (matplotlib.axes.Axes, optional): Axis on which to plot. If None, a new figure is created.
        """
        # If data is a DataStream, extract the specified column.
        if isinstance(data, DataStream):
            df = data.df
            if column is None:
                cols = [col for col in df.columns if col != "time"]
                if not cols:
                    raise ValueError("No valid column found in the DataStream.")
                column = cols[0]
            filtered = df[column].dropna().values
        else:
            filtered = np.array(data).flatten()

        n = len(filtered)
        if n == 0:
            raise ValueError("Data is empty after filtering.")
        
        # Compute ACF values with nlags = int(n / 3)
        nlags = int(n / 3)
        acf_values = acf(filtered, nlags=nlags, fft=False)
        
        # Calculate z-critical and confidence interval
        z_critical = norm.ppf(1 - alpha / 2)
        conf_interval = z_critical / np.sqrt(n)
        
        # Create new figure if no axis is provided.
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the ACF as a stem plot.
        ax.stem(range(len(acf_values)), acf_values, basefmt=" ")
        ax.axhline(conf_interval, color='red', linestyle='--', label=f'95% CI upper: {conf_interval:.3f}')
        ax.axhline(-conf_interval, color='red', linestyle='--', label=f'95% CI lower: {-conf_interval:.3f}')
        ax.set_title(f"ACF of '{column}'")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend()
        
        # Only show the plot if a new figure was created.
        if ax is None:
            plt.show()

    def plot_acf_ensemble(self, ensemble_obj, alpha=0.05, column=None):
        """
        Plot the ACF for each ensemble member individually on a grid of subplots.
        
        The number of rows and columns in the grid is determined based on the number of ensemble members.
        This function loops through each ensemble member (DataStream) in the Ensemble object and calls
        the plot_acf function to generate the individual ACF plots on separate subplots.
        
        Args:
            ensemble_obj (Ensemble): An Ensemble instance containing DataStream members.
            alpha (float): Significance level for the confidence interval (default: 0.05).
            column (str, optional): Column name to use for ACF computation. If None, the first non-'time' column is used.
        """
        n_members = len(ensemble_obj.data_streams)
        # Choose grid dimensions (e.g., up to 3 columns)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        # Flatten axes array for easier iteration.
        axes = np.array(axes).flatten()
        
        for i, ds in enumerate(ensemble_obj.data_streams):
            # For each ensemble member, call plot_acf on its DataStream.
            self.plot_acf(ds, alpha=alpha, column=column, ax=axes[i])
            axes[i].set_title(f"Member {i} ACF")
        
        # Remove any extra subplots if there are fewer members than subplots.
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def ensemble_steady_state_automatic_plot(self, ensemble_obj, variables_to_plot=None, window_size=10, 
                                               start_time=0.0, method="std", threshold=None, robust=True, save=False):
        """
        Plot steady state detection automatically for each ensemble member on a grid.
        
        For each ensemble member in the Ensemble object, for each variable (if multiple are provided,
        all are overlaid on the same subplot), the method uses DataStream.trim() to estimate the steady 
        state start time. If detected, it plots the original signal with:
          - A vertical dashed red line at the estimated steady state start.
          - A horizontal green line at the overall mean (computed from the data after steady state).
          - Shaded regions for ±1, ±2, and ±3 standard deviations.
        If no steady state is detected, it plots the raw signal and prints a message.
        
        The plots are arranged in a grid with one subplot per ensemble member.
        
        Args:
            ensemble_obj (Ensemble): An Ensemble instance.
            variables_to_plot (list, optional): List of variable names to plot. If None, all columns (except 'time')
                from the first member are used.
            window_size (int): Window size for the trim() function.
            start_time (float): Start time for steady state detection.
            method (str): Steady state detection method ('std', 'threshold', or 'rolling_variance').
            threshold (float, optional): Threshold if needed by the method.
            robust (bool): If True, use robust statistics (median/MAD) in the 'std' method.
            save (bool): If True, save the resulting figure to disk.
        """
        # Determine grid dimensions based on number of ensemble members.
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        # For each ensemble member, plot the steady state detection for the chosen variables.
        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.df
            # Determine variables to plot.
            if variables_to_plot is None:
                vars_plot = [col for col in df.columns if col != "time"]
            else:
                vars_plot = [var for var in variables_to_plot if var != "time"]

            ax = axes[i]
            time = df["time"]
            # For overlaying multiple variables on the same subplot, iterate over each variable.
            for var in vars_plot:
                signal = df[var]
                # Create a DataStream copy and attempt automatic trimming.
                ds_temp = DataStream(df)
                trimmed_ds = ds_temp.trim(var, window_size=window_size, start_time=start_time,
                                           method=method, threshold=threshold, robust=robust)
                if trimmed_ds is not None:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()
                    
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    ax.axvline(x=steady_state_start, color='r', linestyle='--', label='SS Start')
                    ax.axhline(y=overall_mean, color='g', linestyle='-', label='Mean')
                    ax.fill_between(time[time >= steady_state_start],
                                    overall_mean - overall_std,
                                    overall_mean + overall_std,
                                    color='blue', alpha=0.3, label='±1 Std')
                    ax.fill_between(time[time >= steady_state_start],
                                    overall_mean - 2 * overall_std,
                                    overall_mean + 2 * overall_std,
                                    color='yellow', alpha=0.2, label='±2 Std')
                    ax.fill_between(time[time >= steady_state_start],
                                    overall_mean - 3 * overall_std,
                                    overall_mean + 3 * overall_std,
                                    color='red', alpha=0.1, label='±3 Std')
                else:
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    print(f"Member {i}: {var} steady state not detected. Plotting raw signal.")
            ax.set_title(f"Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal")
            ax.legend(fontsize="small")
            ax.grid(True)
        
        # Remove any extra subplots.
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(self.output_dir, "ensemble_steady_state_auto.png")
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close(fig)

    def ensemble_steady_state_plot(self, ensemble_obj, variables_to_plot=None, steady_state_start=None, save=False):
        """
        Plot steady state detection for each ensemble member using a user-supplied steady state start.
        
        For each ensemble member in the Ensemble object, the function plots the signal for the specified
        variables (or all non-'time' variables if not provided) and draws:
          - A vertical dashed red line at the user-supplied steady state start.
          - A horizontal green line representing the mean of the data after the steady state.
          - Shaded regions for ±1, ±2, and ±3 standard deviations (computed after the steady state).
        If no steady state start is provided for a variable, the raw signal is plotted and a message is printed.
        
        The plots are arranged in a grid.
        
        Args:
            ensemble_obj (Ensemble): An Ensemble instance.
            variables_to_plot (list, optional): List of variables to plot. If None, all non-'time' columns are used.
            steady_state_start (float or dict, optional): A single float or a dict mapping variable names to steady state start values.
            save (bool): If True, save the resulting figure.
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.df
            if variables_to_plot is None:
                vars_plot = [col for col in df.columns if col != "time"]
            else:
                vars_plot = [var for var in variables_to_plot if var != "time"]
            ax = axes[i]
            time = df["time"]
            for var in vars_plot:
                signal = df[var]
                # Determine steady state start: if steady_state_start is a dict, get value for this var; otherwise, use the float.
                if isinstance(steady_state_start, dict):
                    manual_ss = steady_state_start.get(var, None)
                else:
                    manual_ss = steady_state_start
                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()
                    
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    ax.axvline(x=manual_ss, color='r', linestyle='--', label='SS Start')
                    ax.axhline(y=overall_mean, color='g', linestyle='-', label='Mean')
                    ax.fill_between(time[time >= manual_ss],
                                    overall_mean - overall_std,
                                    overall_mean + overall_std,
                                    color='blue', alpha=0.3, label='±1 Std')
                    ax.fill_between(time[time >= manual_ss],
                                    overall_mean - 2 * overall_std,
                                    overall_mean + 2 * overall_std,
                                    color='yellow', alpha=0.2, label='±2 Std')
                    ax.fill_between(time[time >= manual_ss],
                                    overall_mean - 3 * overall_std,
                                    overall_mean + 3 * overall_std,
                                    color='red', alpha=0.1, label='±3 Std')
                else:
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    print(f"Member {i}: No manual steady state start provided for {var}. Plotting raw signal.")
            ax.set_title(f"Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Signal")
            ax.legend(fontsize="small")
            ax.grid(True)
        
        # Remove extra subplots if any.
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(self.output_dir, "ensemble_steady_state_manual.png")
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close(fig)