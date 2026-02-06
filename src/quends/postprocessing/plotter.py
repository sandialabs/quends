import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

from quends.base.data_stream import DataStream  # Adjust the import if necessary
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
            return {
                f"DataStream {k}": data.data_streams[k].df
                for k in range(len(data))
            }
        elif isinstance(data, dict):
            return data
        else:
            raise ValueError(
                "Input data must be a DataStream instance or a dictionary of DataFrames"
            )

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

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for dataset_name, df in data_frames.items():
            time_series = df["time"]
            num_traces = len(variables_to_plot)
            num_cols = min(5, num_traces)
            num_rows = (num_traces + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)

            if num_traces == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for j, column in enumerate(variables_to_plot):
                axes[j].plot(time_series, df[column], label=column)
                axes[j].set_xlabel("Time")
                axes[j].set_ylabel(f"{column} (Values)")
                axes[j].set_title(f"Time Series of {column}")
                axes[j].legend(fontsize="small")
                axes[j].grid(True)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Time Series Plots for {self.format_dataset_name(dataset_name)}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"time_series_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=350)
            plt.show()
            plt.close()

        return axes

    def trace_plot_with_mean(self, data, variables_to_plot=None, save=False):
        """
        Plot individual (trace) time series data from a DataStream or a dictionary of DataFrames.
        The resulting plots are displayed and optionally saved if 'save' is True.
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for dataset_name, df in data_frames.items():
            time_series = df["time"]
            num_traces = len(variables_to_plot)
            num_cols = min(5, num_traces)
            num_rows = (num_traces + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)

            if num_traces == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for j, column in enumerate(variables_to_plot):
                ds = DataStream(df)
                tds = ds.trim(column)
                m = tds.mean()[column]["mean"]
                lower, upper = tds.confidence_interval()[column][
                    "confidence interval"
                ]
                axes[j].plot(time_series, df[column], label=column)
                axes[j].plot(
                    tds.df["time"],
                    m * np.ones(len(tds.df["time"])),
                    color="r",
                    label="Mean",
                )
                axes[j].fill_between(
                    tds.df["time"],
                    lower,
                    upper,
                    color="r",
                    alpha=0.2,
                    label="Uncertainty",
                )
                axes[j].set_xlabel("Time")
                axes[j].set_ylabel(f"{column} (Values)")
                axes[j].set_title(f"Time Series of {column}")
                axes[j].legend(fontsize="small")
                axes[j].grid(True)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Time Series Plots for {self.format_dataset_name(dataset_name)}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"time_series_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=350)
            plt.show()
            plt.close()

    def ensemble_trace_plot(self, data, variables_to_plot=None, save=False):
        """
        Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.
        The resulting plots are displayed and optionally saved if 'save' is True.
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for var in variables_to_plot:
            plt.figure(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                plt.plot(df["time"], df[var], label=dataset_name)
            plt.xlabel("Time")
            plt.ylabel(f"{var} (Values)")
            plt.title(f"Ensemble Time Series of {var}")
            plt.legend()
            plt.tight_layout()
            if save:
                save_path = os.path.join(
                    self.output_dir, f"ensemble_time_series_{var}.png"
                )
                plt.savefig(save_path, dpi=350)
            plt.show()
            plt.close()

    def ensemble_trace_plot_with_mean(
        self, data, variables_to_plot=None, save=False
    ):
        """
        Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.
        The resulting plots are displayed and optionally saved if 'save' is True.
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for var in variables_to_plot:
            plt.figure(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                plt.plot(df["time"], df[var], label=dataset_name)
            plt.xlabel("Time")
            plt.ylabel(f"{var} (Values)")
            plt.title(f"Ensemble Time Series of {var}")
            plt.legend()
            plt.tight_layout()
            if save:
                save_path = os.path.join(
                    self.output_dir, f"ensemble_time_series_{var}.png"
                )
                plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()

    def steady_state_automatic_plot_flexx(
        self,
        data,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        save=False,
    ):
        """
        Plot steady state detection for each variable in the data. For each variable, the method uses the
        DataStream.trim() function to estimate the steady state start time. If a steady state is detected,
        the function plots the original time series along with:
            - A vertical dashed red line indicating the steady state start time.
            - A horizontal green line at the overall mean (computed from data after the steady state start).
            - Shaded regions representing ±1, ±2, and ±3 standard deviations.
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = [axes] if num_vars == 1 else axes.flatten()

            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]

                ds = DataStream(df)
                trimmed_ds = ds.trim(
                    column,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )

                if trimmed_ds is not None:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=steady_state_start,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        y=overall_mean, color="g", linestyle="-", label="Mean"
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - overall_std,
                        overall_mean + overall_std,
                        color="blue",
                        alpha=0.3,
                        label="1 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - 2 * overall_std,
                        overall_mean + 2 * overall_std,
                        color="yellow",
                        alpha=0.2,
                        label="2 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - 3 * overall_std,
                        overall_mean + 3 * overall_std,
                        color="red",
                        alpha=0.1,
                        label="3 Std Dev",
                    )
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"{column} is stationary but steady state not achieved. Run longer."
                    )

                ax.set_title(f"Steady-State Detection of {column}")
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{column} (Values)")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_detection_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close(fig)

    def steady_state_plot(
        self, data, variables_to_plot=None, steady_state_start=None, save=False
    ):
        """
        Plot steady state detection for each variable in the data using a user-supplied steady state start.
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = [axes] if num_vars == 1 else axes.flatten()

            for j, column in enumerate(variables_to_plot):
                ax = axes[j]
                time = df["time"]
                signal = df[column]

                if isinstance(steady_state_start, dict):
                    manual_ss = steady_state_start.get(column, None)
                else:
                    manual_ss = steady_state_start

                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=manual_ss,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        y=overall_mean, color="g", linestyle="-", label="Mean"
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        overall_mean - overall_std,
                        overall_mean + overall_std,
                        color="blue",
                        alpha=0.3,
                        label="1 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        overall_mean - 2 * overall_std,
                        overall_mean + 2 * overall_std,
                        color="yellow",
                        alpha=0.2,
                        label="2 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        overall_mean - 3 * overall_std,
                        overall_mean + 3 * overall_std,
                        color="red",
                        alpha=0.1,
                        label="3 Std Dev",
                    )
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"For {column}, no manual steady state start provided. Plotting raw signal."
                    )

                ax.set_title(f"Manual Steady-State Plot of {column}")
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{column} (Values)")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_manual_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=350)
            plt.show()
            plt.close(fig)

    def plot_acf(self, data, alpha=0.05, column=None, ax=None):
        """
        Plot the Autocorrelation Function (ACF) for a given data stream or array-like object.
        """
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

        nlags = int(n / 3)
        acf_values = acf(filtered, nlags=nlags, fft=False)
        z_critical = norm.ppf(1 - alpha / 2)
        conf_interval = z_critical / np.sqrt(n)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        ax.stem(range(len(acf_values)), acf_values, basefmt=" ")
        ax.axhline(
            conf_interval,
            color="red",
            linestyle="--",
            label=f"95% CI upper {conf_interval:.3f}",
        )
        ax.axhline(
            -conf_interval,
            color="red",
            linestyle="--",
            label=f"95% CI lower {-conf_interval:.3f}",
        )
        ax.set_title(f"Autocorrelation Function (ACF) of {column}")
        ax.set_xlabel("Lag")
        ax.set_ylabel(f"{column} (ACF Values)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if ax is None:
            plt.show()

    def plot_acf_ensemble(self, ensemble_obj, alpha=0.05, column=None):
        """
        Plot the ACF for each ensemble member individually on a grid of subplots.
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            self.plot_acf(ds, alpha=alpha, column=column, ax=axes[i])
            axes[i].set_title(f"Member {i} ACF")
            axes[i].set_xlabel("Lag")
            axes[i].set_ylabel(f"{column if column else 'Signal'} (ACF Values)")
            axes[i].grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def ensemble_steady_state_automatic_plot_flexx(
        self,
        ensemble_obj,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        save=False,
    ):
        """
        Plot steady state detection automatically for each ensemble member on a grid.
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(math.ceil(n_members / ncols))
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
                ds_temp = DataStream(df)
                trimmed_ds = ds_temp.trim(
                    var,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )
                if trimmed_ds is not None and not trimmed_ds.df.empty:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    ax.axvline(
                        x=steady_state_start,
                        color="r",
                        linestyle="--",
                        label="SS Start",
                    )
                    ax.axhline(
                        y=overall_mean, color="g", linestyle="-", label="Mean"
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - overall_std,
                        overall_mean + overall_std,
                        color="blue",
                        alpha=0.3,
                        label="±1 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - 2 * overall_std,
                        overall_mean + 2 * overall_std,
                        color="yellow",
                        alpha=0.2,
                        label="±2 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - 3 * overall_std,
                        overall_mean + 3 * overall_std,
                        color="red",
                        alpha=0.1,
                        label="±3 Std Dev",
                    )
                else:
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    print(f"Member {i}: {var} steady state not detected.")

            ax.set_title(f"Steady-State Detection for Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{', '.join(vars_plot)} (Values)")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(
                self.output_dir, "ensemble_steady_state_auto.png"
            )
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close(fig)

    def ensemble_steady_state_plot(
        self,
        ensemble_obj,
        variables_to_plot=None,
        steady_state_start=None,
        save=False,
    ):
        """
        Plot steady state detection for each ensemble member using a user-supplied steady state start.
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
                if isinstance(steady_state_start, dict):
                    manual_ss = steady_state_start.get(var, None)
                else:
                    manual_ss = steady_state_start
                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    ax.axvline(
                        x=manual_ss, color="r", linestyle="--", label="SS Start"
                    )
                    ax.axhline(
                        y=overall_mean, color="g", linestyle="-", label="Mean"
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        overall_mean - overall_std,
                        overall_mean + overall_std,
                        color="blue",
                        alpha=0.3,
                        label="±1 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        overall_mean - 2 * overall_std,
                        overall_mean + 2 * overall_std,
                        color="yellow",
                        alpha=0.2,
                        label="±2 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        overall_mean - 3 * overall_std,
                        overall_mean + 3 * overall_std,
                        color="red",
                        alpha=0.1,
                        label="±3 Std Dev",
                    )
                else:
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    print(
                        f"Member {i}: No manual steady state start for {var}."
                    )

            ax.set_title(f"Manual Steady-State Plot for Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{', '.join(vars_plot)} (Values)")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(
                self.output_dir, "ensemble_steady_state_manual.png"
            )
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close(fig)

    def plot_ensemble_old_new(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
        condensed_legend=False,
    ):
        """
        Plot each ensemble member together with the ensemble average.

        Enhancements:
        -------------
        • If only one variable is plotted, the legend is positioned directly under
          the plot (closer to the x-axis).
        • If multiple variables are plotted, the legend remains centered below but
          with reduced vertical spacing.
        • If `condensed_legend=True`, all ensemble member traces are grouped into
          one legend entry labeled 'Individual Members' and the ensemble mean is
          shown as 'Ensemble Average'.

        Parameters
        ----------
        ensemble_obj : Ensemble
            Ensemble object containing multiple DataStreams.
        variables_to_plot : list, optional
            Variables to plot (default: all except 'time').
        show_plots : bool, default=False
            If True, show figures interactively.
        save : bool, default=False
            If True, save figures to the output directory.
        condensed_legend : bool, default=False
            If True, replace all individual member labels with one unified legend entry.
        """
        member_dfs = {
            f"Member {i}": ds.df
            for i, ds in enumerate(ensemble_obj.data_streams)
        }

        avg_stream = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_stream.df

        all_cols = avg_stream.df.columns.tolist()
        vars_to_plot = (
            [c for c in all_cols if c != "time"]
            if variables_to_plot is None
            else [c for c in variables_to_plot if c != "time"]
        )

        if not vars_to_plot:
            raise ValueError("No variables to plot.")
        n_vars = len(vars_to_plot)

        # === Create subplots ===
        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            first_drawn = False

            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"],
                        df[var],
                        label="Ensemble Average",
                        color="black",
                        linewidth=2.5,
                        zorder=5,
                    )
                else:
                    # For condensed legend, only label once
                    if condensed_legend:
                        label = (
                            "Individual Members" if not first_drawn else None
                        )
                    else:
                        label = name
                    ax.plot(
                        df["time"],
                        df[var],
                        label=label,
                        alpha=0.3,
                        linewidth=1.0,
                    )
                    first_drawn = True

            ax.set_title(f"Ensemble Average and Members of {var}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{var} (Values)")
            ax.grid(True, alpha=0.3)

        # Remove any extra subplot slots
        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        # === Legend logic ===
        handles, labels = axes[0].get_legend_handles_labels()
        if condensed_legend:
            # Keep only "Individual Members" and "Ensemble Average"
            unique = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen and l is not None:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique)

        legend_ncol = min(len(labels), 4)
        legend_nrow = math.ceil(len(labels) / legend_ncol)

        # Adjust vertical legend placement depending on variable count
        if n_vars == 1:
            legend_y = -0.05  # closer to single plot
        else:
            legend_y = -0.08  # tighter than before for multi-panel
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_ncol,
            fontsize="small",
            frameon=False,
        )

        # Adjust spacing
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        # === Save/Show ===
        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_members_and_average.png"
            )
            fig.savefig(outpath, dpi=350)
            print(f"✅ Saved ensemble plot to {outpath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_ensemble_good(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
        condensed_legend=False,
        y_range=None,  # <--- NEW PARAMETER
    ):
        """
        Plot each ensemble member together with the ensemble average.

        Enhancements:
        -------------
        • If only one variable is plotted, the legend is aligned directly beneath
          the x-axis (flush left under the main plot) with minimal vertical spacing.
        • If multiple variables are plotted, the legend remains centered below
          all subplots but with reduced white space.
        • If `condensed_legend=True`, all ensemble member traces are grouped into
          a single legend entry labeled 'Individual Members' while the ensemble
          mean retains its label 'Ensemble Average'.

        Parameters
        ----------
        ensemble_obj : Ensemble
            Ensemble object containing multiple DataStreams.
        variables_to_plot : list, optional
            Variables to plot (default: all except 'time').
        show_plots : bool, default=False
            If True, display figures interactively.
        save : bool, default=False
            If True, export figures to the output directory.
        condensed_legend : bool, default=False
            If True, show only two legend entries:
            one for the ensemble average and one for all individual members.
        """
        member_dfs = {
            f"Member {i}": ds.df
            for i, ds in enumerate(ensemble_obj.data_streams)
        }
        avg_stream = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_stream.df

        all_cols = avg_stream.df.columns.tolist()
        vars_to_plot = (
            [c for c in all_cols if c != "time"]
            if variables_to_plot is None
            else [c for c in variables_to_plot if c != "time"]
        )
        if not vars_to_plot:
            raise ValueError("No variables to plot.")
        n_vars = len(vars_to_plot)

        # === Create subplots ===
        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            first_drawn = False
            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"],
                        df[var],
                        label="Ensemble Average",
                        color="black",
                        linewidth=2.5,
                        zorder=5,
                    )
                else:
                    # For condensed legend, label once only
                    if condensed_legend:
                        label = (
                            "Individual Members" if not first_drawn else None
                        )
                    else:
                        label = name
                    ax.plot(
                        df["time"],
                        df[var],
                        label=label,
                        alpha=0.3,
                        linewidth=1.0,
                    )
                    first_drawn = True

            # --- Titles and axes styling ---
            ax.set_title(
                f"Ensemble Average and Members of {var}",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel("Time", fontsize=14, fontweight="bold")
            ax.set_ylabel(f"{var} (Values)", fontsize=14, fontweight="bold")
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(True, alpha=0.3)

            # >>> NEW: enforce manual y-axis limits if specified
            if y_range is not None:
                ax.set_ylim(y_range)

            # ax.set_title(f"Ensemble Average and Members of {var}")
            # ax.set_xlabel("Time")
            # ax.set_ylabel(f"{var} (Values)")
            # ax.grid(True, alpha=0.3)

        # Remove unused subplot slots
        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        # === Legend handling ===
        handles, labels = axes[0].get_legend_handles_labels()

        if condensed_legend:
            # Keep only "Individual Members" + "Ensemble Average"
            unique = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen and l is not None:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique)

        legend_ncol = min(len(labels), 4)

        # --- Legend placement ---

        if n_vars == 1:
            # --- Single-variable layout: legend left-aligned under x-axis ---
            ax = axes[0]
            legend = ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0, -0.10),
                ncol=legend_ncol,
                fontsize=11,
                # fontsize="small",
                frameon=False,
                handlelength=2.5,
                columnspacing=1.5,
            )
            # Reduce padding between plot and legend
            plt.subplots_adjust(bottom=0.15)
        else:
            # --- Multi-variable layout: shared centered legend ---
            legend_y = -0.08
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=legend_ncol,
                fontsize="small",
                frameon=False,
            )
            plt.tight_layout(rect=[0, 0.05, 1, 1])

        # === Save/Show ===
        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_members_and_average.png"
            )
            fig.savefig(outpath, dpi=350, bbox_inches="tight")
            print(f"✅ Saved ensemble plot to {outpath}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_ensemble_old(
        self, ensemble_obj, variables_to_plot=None, show_plots=False, save=False
    ):
        """
        Plot each ensemble member together with the ensemble average.
        """
        member_dfs = {
            f"Member {i}": ds.df
            for i, ds in enumerate(ensemble_obj.data_streams)
        }

        avg_stream = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_stream.df

        all_cols = avg_stream.df.columns.tolist()
        vars_to_plot = (
            [c for c in all_cols if c != "time"]
            if variables_to_plot is None
            else [c for c in variables_to_plot if c != "time"]
        )
        if not vars_to_plot:
            raise ValueError("No variables to plot.")
        n_vars = len(vars_to_plot)

        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"],
                        df[var],
                        label=name,
                        color="black",
                        linewidth=2.5,
                        zorder=5,
                    )
                else:
                    ax.plot(
                        df["time"],
                        df[var],
                        label=name,
                        alpha=0.3,
                        linewidth=1.0,
                    )
            ax.set_title(f"Ensemble Average and Members of {var}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{var} (Values)")
            ax.grid(True, alpha=0.3)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        legend_ncol = min(len(labels), 4)
        legend_nrow = math.ceil(len(labels) / legend_ncol)
        legend_y = -0.05 * legend_nrow
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_ncol,
            fontsize="small",
            frameon=False,
        )

        bottom_margin = 0.005 * 1 + 0.002
        plt.tight_layout(rect=[0, bottom_margin, 1, 1])

        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_members_and_average.png"
            )
            fig.savefig(outpath, dpi=350)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def steady_state_automatic_plot_with_stats_flexx(
        self,
        data,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        save=False,
    ):
        """
        Plot steady state detection for each variable in the data, annotated with statistics.
        """
        data_frames = self._prepare_data_frames(data)
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = axes.flatten() if num_vars > 1 else [axes]

            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]

                ds = DataStream(df)
                trimmed_ds = ds.trim(
                    column_name=column,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )

                if trimmed_ds is not None and not trimmed_ds.df.empty:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        steady_state_start,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        overall_mean, color="g", linestyle="-", label="Mean"
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - overall_std,
                        overall_mean + overall_std,
                        color="blue",
                        alpha=0.3,
                        label="±1 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - 2 * overall_std,
                        overall_mean + 2 * overall_std,
                        color="yellow",
                        alpha=0.2,
                        label="±2 Std Dev",
                    )
                    ax.fill_between(
                        time[time >= steady_state_start],
                        overall_mean - 3 * overall_std,
                        overall_mean + 3 * overall_std,
                        color="red",
                        alpha=0.1,
                        label="±3 Std Dev",
                    )

                    # Annotate computed stats
                    stats = trimmed_ds.compute_statistics(
                        column_name=column
                    )["results"][column]
                    ci_low, ci_high = stats["confidence_interval"]
                    pm_low, pm_high = stats["pm_std"]
                    txt = (
                        f"μ = {stats['mean']:.3f}\n"
                        f"SEM = {stats['mean_uncertainty']:.3f}\n"
                        f"CI = [{ci_low:.3f}, {ci_high:.3f}]\n"
                        f"±STD = [{pm_low:.3f}, {pm_high:.3f}]"
                    )
                    ax.text(
                        0.03,
                        0.97,
                        txt,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize="small",
                        bbox=dict(
                            facecolor="white", alpha=0.6, edgecolor="gray"
                        ),
                    )

                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"{column}: No steady state detected or insufficient data."
                    )

                ax.set_title(f"Steady-State Detection with Stats: {column}")
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{column} (Values)")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            for k in range(num_vars, len(axes)):
                fig.delaxes(axes[k])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_with_stats_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close(fig)

    def plot_block_acf(
        self, test_result, column=None, ax=None, alpha=0.05, title_prefix=""
    ):
        """
        Plot the block autocorrelation function (ACF) from test_block_independence() results.
        """
        if (
            isinstance(test_result, dict)
            and column is None
            and any(
                isinstance(v, dict) and "acf" in v for v in test_result.values()
            )
        ):
            raise ValueError(
                "Input looks like a multi-column output. Please specify `column` to plot."
            )
        if (
            column is not None
            and isinstance(test_result, dict)
            and column in test_result
        ):
            test_result = test_result[column]

        acf_vals = np.asarray(test_result.get("acf", []))
        confint = test_result.get("acf_confint", None)
        n_lags = len(acf_vals)
        pval = test_result.get("ljungbox_pvalue", None)
        indep = test_result.get("independent", None)
        win = test_result.get("window_size", None) or test_result.get(
            "est_win", None
        )
        msg = test_result.get("message", "")
        colname = (
            column
            if column is not None
            else test_result.get("column_name", None)
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        markerline, stemlines, baseline = ax.stem(
            range(n_lags), acf_vals, basefmt=" "
        )
        ax.set_ylim(
            min(-1, np.min(acf_vals) - 0.1), max(1, np.max(acf_vals) + 0.1)
        )

        if confint is not None and np.isfinite(confint):
            ax.axhline(+confint, ls="--", c="grey", lw=1)
            ax.axhline(-confint, ls="--", c="grey", lw=1)
            ax.fill_between(
                range(n_lags),
                -confint,
                confint,
                color="grey",
                alpha=0.1,
                zorder=0,
            )

        title = f"{title_prefix}Block Means ACF"
        if colname:
            title += f" [{colname}]"
        if win is not None:
            title += f" (win={win})"
        if pval is not None:
            title += (
                f"\nLjung-Box p={pval:.3g} ({'INDEP' if indep else 'AUTO'})"
            )
        if msg:
            title += "\n" + msg

        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel(f"{colname if colname else 'Signal'} (Block ACF Values)")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_ensemble_block_acf(
        self,
        ensemble_results,
        columns=None,
        alpha=0.05,
        sharey=True,
        suptitle="Ensemble Block Means ACF",
    ):
        """
        Plot block ACFs for all members/columns in ensemble_results.
        """
        if columns is None:
            columns = list(ensemble_results.keys())
        n = len(columns)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=sharey
        )
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(columns):
            ax = axes[i]
            self.plot_block_acf(
                ensemble_results, column=col, ax=ax, alpha=alpha
            )
            ax.set_xlabel("Lag")
            ax.set_ylabel(f"{col} (Block ACF Values)")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(suptitle)
        return fig, axes

    def steady_state_automatic_plot_good(
        self,
        data,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        show_std_bands=False,
        save=False,
        y_range=None,  # <--- NEW PARAMETER
    ):
        """
        Plot steady-state detection for each variable in the data, with flexible control
        over whether to display ±1, ±2, and ±3 standard-deviation bands.

        This function is identical in structure to steady_state_automatic_plot() but
        introduces the flag ``show_std_bands`` to allow optional shading of standard
        deviation regions even when the trimming method is not 'std'.  This enables a
        unified plotting style across 'std', 'threshold', and 'rolling_variance' methods.

        For each variable, the function performs:
            • Automatic steady-state detection using DataStream.trim().
            • A vertical dashed red line at the detected steady-state start.
            • A horizontal green line showing the mean of data after the steady-state start.
            • Optionally, shaded regions representing ±1, ±2, and ±3 standard deviations.

        Parameters
        ----------
        data : DataStream or dict
            Input time-series data.
        variables_to_plot : list, optional
            Variables (columns) to visualize. If None, all numeric variables except 'time' are used.
        batch_size : int, default=10
            Sliding window size for stability detection.
        start_time : float, default=0.0
            Time offset to ignore initial transient region.
        method : str, {'std', 'threshold', 'rolling_variance'}, default='std'
            Method used for steady-state detection.
        threshold : float, optional
            Threshold value for 'threshold' method.
        robust : bool, default=True
            Whether to use robust (median/MAD) statistics.
        show_std_bands : bool, default=True
            Whether to plot ±1, ±2, ±3 σ shaded regions even for non-'std' methods.
        save : bool, default=False
            If True, export the figure at 300 DPI to self.output_dir.

        Notes
        -----
        When ``method == 'std'``, the ±σ bands are always drawn.
        When ``method != 'std'``, the shaded regions are drawn only if
        ``show_std_bands=True``; otherwise, only mean and detection lines are shown.
        """

        # --- Prepare data frames ---
        data_frames = self._prepare_data_frames(data)

        # Determine which variables to plot
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        # --- Loop through each dataset ---
        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = [axes] if num_vars == 1 else axes.flatten()

            # --- Plot each variable ---
            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]

                # Trim to detect steady-state region
                ds = DataStream(df)
                trimmed_ds = ds.trim(
                    column,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )

                # --- If a steady-state region is found ---
                if trimmed_ds is not None and not trimmed_ds.df.empty:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    # Core plot: signal trace and indicators
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=steady_state_start,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        y=overall_mean,
                        color="g",
                        linestyle="-",
                        label="Mean (Post-SS)",
                    )

                    # --- Conditional shading logic ---
                    if method == "std" or show_std_bands:
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - overall_std,
                            overall_mean + overall_std,
                            color="blue",
                            alpha=0.3,
                            label="±1 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 2 * overall_std,
                            overall_mean + 2 * overall_std,
                            color="yellow",
                            alpha=0.2,
                            label="±2 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 3 * overall_std,
                            overall_mean + 3 * overall_std,
                            color="red",
                            alpha=0.1,
                            label="±3 Std Dev",
                        )

                # --- If no steady-state detected ---
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"[⚠] {column}: steady state not detected. Consider longer run."
                    )

                # --- Styling and formatting ---
                ax.set_title(
                    f"Steady-State Detection (Flex) of {column}",
                    fontsize=16,
                    fontweight="bold",
                )
                ax.set_xlabel("Time", fontsize=14, fontweight="bold")
                ax.set_ylabel(
                    f"{column} (Values)", fontsize=14, fontweight="bold"
                )
                ax.tick_params(axis="both", labelsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)

                # >>> NEW: manually fix y-axis limits if desired
                if y_range is not None:
                    ax.set_ylim(y_range)

                # --- Axis formatting ---
                # ax.set_title(f"Steady-State Detection (Flex) of {column}")
                # ax.set_xlabel("Time")
                # ax.set_ylabel(f"{column} (Values)")
                # ax.legend(fontsize="small")
                # ax.grid(True, alpha=0.3)

            # Remove unused subplot axes
            for k in range(len(variables_to_plot), len(axes)):
                fig.delaxes(axes[k])

            # Layout and optional save
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_auto_flex_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=300)
                print(f"✅ Saved flexible steady-state plot to {save_path}")
            plt.show()
            plt.close(fig)

    def steady_state_plot(
        self,
        data,
        variables_to_plot=None,
        steady_state_start=None,
        method="std",
        show_std_bands=False,
        save=False,
    ):
        """
        Plot steady state detection for each variable in the data using a user-supplied
        steady-state start, with flexible control over ±1/±2/±3 standard-deviation shading.

        This method mirrors steady_state_plot() but introduces two controls:
            • ``method``: a label to indicate which trimming rationale produced the supplied
              steady-state start (e.g., 'std', 'threshold', or 'rolling_variance').
            • ``show_std_bands``: if True, draw ±1/±2/±3 σ shaded regions even when the
              trimming rationale was not 'std'. When False and method != 'std', only the
              mean (post-SS) line and steady-state marker are shown.

        For each variable:
            • The full signal is plotted.
            • A vertical dashed red line indicates the provided steady-state start.
            • A horizontal green line shows the mean of the data after steady state.
            • Optionally, shaded regions mark ±1, ±2, and ±3 standard deviations (post-SS).

        Parameters
        ----------
        data : DataStream or dict
            Input time-series data.
        variables_to_plot : list, optional
            Variables (columns) to visualize. If None, all numeric variables except 'time' are used.
        steady_state_start : float or dict, optional
            Single float applied to all variables, or a dict mapping variable → start time.
        method : str, {'std', 'threshold', 'rolling_variance'}, default='std'
            Descriptive tag for how the steady-state start was obtained (controls band logic).
        show_std_bands : bool, default=True
            Whether to draw ±1/±2/±3 σ bands even if method != 'std'.
        save : bool, default=False
            If True, export the figure at 350 DPI to self.output_dir.
        """
        data_frames = self._prepare_data_frames(data)

        # Determine variables to plot
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        # Iterate over datasets
        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = [axes] if num_vars == 1 else axes.flatten()

            for j, column in enumerate(variables_to_plot):
                ax = axes[j]
                time = df["time"]
                signal = df[column]

                # Resolve manual steady-state start for this variable
                if isinstance(steady_state_start, dict):
                    manual_ss = steady_state_start.get(column, None)
                else:
                    manual_ss = steady_state_start

                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    # Core plot
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=manual_ss,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        y=overall_mean,
                        color="g",
                        linestyle="-",
                        label="Mean (Post-SS)",
                    )

                    # Conditional shaded bands
                    if method == "std" or show_std_bands:
                        ax.fill_between(
                            time[time >= manual_ss],
                            overall_mean - overall_std,
                            overall_mean + overall_std,
                            color="blue",
                            alpha=0.3,
                            label="±1 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= manual_ss],
                            overall_mean - 2 * overall_std,
                            overall_mean + 2 * overall_std,
                            color="yellow",
                            alpha=0.2,
                            label="±2 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= manual_ss],
                            overall_mean - 3 * overall_std,
                            overall_mean + 3 * overall_std,
                            color="red",
                            alpha=0.1,
                            label="±3 Std Dev",
                        )
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"For {column}, no manual steady-state start provided. Plotting raw signal."
                    )

                ax.set_title(f"Manual Steady-State (Flex) of {column}")
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{column} (Values)")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            # Clean empty axes
            for k in range(len(variables_to_plot), len(axes)):
                fig.delaxes(axes[k])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_manual_flex_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=350)
                print(f"✅ Saved manual steady-state flex plot to {save_path}")
            plt.show()
            plt.close(fig)

    def ensemble_steady_state_plot(
        self,
        ensemble_obj,
        variables_to_plot=None,
        steady_state_start=None,
        method="std",
        show_std_bands=False,
        save=False,
    ):
        """
        Plot steady-state detection for each ensemble member, with flexible control over
        whether to display ±1/±2/±3 standard-deviation bands.

        This function extends ensemble_steady_state_plot() by adding the parameter
        ``show_std_bands`` that allows consistent visualization of shaded regions even
        when the steady-state detection was derived from non-'std' methods such as
        'threshold' or 'rolling_variance'.

        For each ensemble member:
            • The full signal is plotted.
            • A vertical dashed red line marks the steady-state start (user-specified or dict).
            • A horizontal green line shows the mean of data after steady state.
            • Shaded regions (±1, ±2, ±3 σ) are conditionally drawn based on the
              method and flag combination.

        Parameters
        ----------
        ensemble_obj : Ensemble
            An Ensemble instance containing DataStream members.
        variables_to_plot : list, optional
            List of variable names to plot. If None, all columns (except 'time') are used.
        steady_state_start : float or dict, optional
            Single float or dict mapping variable names to steady-state start values.
        method : str, {'std', 'threshold', 'rolling_variance'}, default='std'
            Descriptive tag for the steady-state detection approach used.
        show_std_bands : bool, default=True
            If True, draw ±1/±2/±3 σ shaded regions for all methods.
            If False and method != 'std', shaded regions are omitted.
        save : bool, default=False
            If True, export the resulting figure at 300 DPI to self.output_dir.
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        # Loop over ensemble members
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

                # Determine steady-state start (float or dict)
                if isinstance(steady_state_start, dict):
                    manual_ss = steady_state_start.get(var, None)
                else:
                    manual_ss = steady_state_start

                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    ax.axvline(
                        x=manual_ss,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        y=overall_mean,
                        color="g",
                        linestyle="-",
                        label="Mean (Post-SS)",
                    )

                    # Conditional shading logic
                    if method == "std" or show_std_bands:
                        ax.fill_between(
                            time[time >= manual_ss],
                            overall_mean - overall_std,
                            overall_mean + overall_std,
                            color="blue",
                            alpha=0.3,
                            label="±1 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= manual_ss],
                            overall_mean - 2 * overall_std,
                            overall_mean + 2 * overall_std,
                            color="yellow",
                            alpha=0.2,
                            label="±2 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= manual_ss],
                            overall_mean - 3 * overall_std,
                            overall_mean + 3 * overall_std,
                            color="red",
                            alpha=0.1,
                            label="±3 Std Dev",
                        )
                else:
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    print(
                        f"Member {i}: No manual steady-state start provided for {var}."
                    )

            # Axis formatting
            ax.set_title(f"Steady-State Detection (Flex) — Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{', '.join(vars_plot)} (Values)")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        # Remove extra subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(
                self.output_dir, "ensemble_steady_state_flex.png"
            )
            plt.savefig(save_path, dpi=300)
            print(
                f"✅ Saved flexible ensemble steady-state figure to {save_path}"
            )
        plt.show()
        plt.close(fig)

    def ensemble_steady_state_automatic_plot(
        self,
        ensemble_obj,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        show_std_bands=False,
        save=False,
    ):
        """
        Plot steady-state detection automatically for each ensemble member with flexible control
        over the display of ±1/±2/±3 standard-deviation bands.

        This function extends `ensemble_steady_state_automatic_plot()` by introducing the
        parameters `method` and `show_std_bands`, allowing users to visualize steady-state
        results consistently across all detection methods (standard deviation, threshold, or
        rolling variance).

        For each ensemble member:
            • The raw signal is plotted.
            • A vertical dashed red line indicates the steady-state start.
            • A horizontal green line marks the post-SS mean value.
            • Shaded regions representing ±1, ±2, and ±3 σ are drawn conditionally:
                - Always for method == 'std'.
                - Optionally for other methods if `show_std_bands=True`.

        Parameters
        ----------
        ensemble_obj : Ensemble
            Ensemble instance containing multiple DataStream objects.
        variables_to_plot : list, optional
            List of variable names to plot. If None, all columns (except 'time') are used.
        batch_size : int, default=10
            Sliding window size used in the steady-state detection algorithm.
        start_time : float, default=0.0
            Time to ignore before detection begins.
        method : str, {'std', 'threshold', 'rolling_variance'}, default='std'
            Steady-state detection method used. Determines whether shaded bands are drawn.
        threshold : float, optional
            Threshold parameter for the 'threshold' method.
        robust : bool, default=True
            Whether to use median/MAD instead of mean/std for robust stability estimation.
        show_std_bands : bool, default=True
            Whether to show ±σ bands when the method is not 'std'.
        save : bool, default=False
            If True, saves the figure to the output directory at 300 DPI.
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(math.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        # --- Iterate over each ensemble member ---
        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.df
            if variables_to_plot is None:
                vars_plot = [col for col in df.columns if col != "time"]
            else:
                vars_plot = [var for var in variables_to_plot if var != "time"]

            ax = axes[i]
            time = df["time"]

            # --- Loop through all variables for this member ---
            for var in vars_plot:
                signal = df[var]
                ds_temp = DataStream(df)
                trimmed_ds = ds_temp.trim(
                    var,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )

                if trimmed_ds is not None and not trimmed_ds.df.empty:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    # Main signal and steady-state indicators
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    ax.axvline(
                        x=steady_state_start,
                        color="r",
                        linestyle="--",
                        label="SS Start",
                    )
                    ax.axhline(
                        y=overall_mean,
                        color="g",
                        linestyle="-",
                        label="Mean (Post-SS)",
                    )

                    # --- Conditional shaded region logic ---
                    if method == "std" or show_std_bands:
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - overall_std,
                            overall_mean + overall_std,
                            color="blue",
                            alpha=0.3,
                            label="±1 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 2 * overall_std,
                            overall_mean + 2 * overall_std,
                            color="yellow",
                            alpha=0.2,
                            label="±2 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 3 * overall_std,
                            overall_mean + 3 * overall_std,
                            color="red",
                            alpha=0.1,
                            label="±3 Std Dev",
                        )

                else:
                    ax.plot(time, signal, label=f"{var}", alpha=0.7)
                    print(
                        f"[⚠] Member {i}: {var} steady state not detected or insufficient data."
                    )

            # --- Axis formatting ---
            ax.set_title(f"Steady-State Detection (Flex) — Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{', '.join(vars_plot)} (Values)")
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        # Remove unused subplot axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # --- Layout and Save ---
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(
                self.output_dir, "ensemble_steady_state_auto_flex.png"
            )
            plt.savefig(save_path, dpi=300)
            print(
                f"✅ Flexible ensemble steady-state plot saved to {save_path}"
            )
        plt.show()
        plt.close(fig)

    def steady_state_automatic_plot_with_stats(
        self,
        data,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        show_std_bands=False,
        save=False,
    ):
        """
        Plot steady-state detection for each variable in the data, annotated with key
        statistical metrics, and with flexible control over the display of ±1/±2/±3
        standard-deviation bands.

        This function is equivalent in structure to steady_state_automatic_plot_with_stats(),
        but introduces two parameters—``method`` and ``show_std_bands``—to unify visualization
        across different steady-state detection approaches ('std', 'threshold', and
        'rolling_variance').

        For each variable, the following are plotted:
            • The original time series signal.
            • A vertical dashed red line at the estimated steady-state start.
            • A horizontal green line representing the mean of the post-SS data.
            • Optionally, shaded regions representing ±1, ±2, and ±3 σ around the mean.
            • A text box summarizing key descriptive statistics (μ, SEM, CI, ±STD).

        Parameters
        ----------
        data : DataStream or dict
            Input time-series data.
        variables_to_plot : list, optional
            List of variable names to visualize. If None, all columns (except 'time') are used.
        batch_size : int, default=10
            Sliding window size used in steady-state detection.
        start_time : float, default=0.0
            Time to ignore before attempting steady-state detection.
        method : str, {'std', 'threshold', 'rolling_variance'}, default='std'
            Trimming method label; determines shading behavior.
        threshold : float, optional
            Threshold value for 'threshold' method.
        robust : bool, default=True
            Whether to use median/MAD instead of mean/std when applicable.
        show_std_bands : bool, default=True
            Whether to plot ±1/±2/±3 σ shaded regions when method != 'std'.
        save : bool, default=False
            If True, export the resulting figure at 300 DPI to self.output_dir.

        Notes
        -----
        • When ``method == 'std'``, shaded bands are always drawn.
        • When ``method != 'std'``, shaded regions appear only if ``show_std_bands=True``.
        • Statistical annotations are computed from DataStream.compute_statistics().
        """

        # --- Prepare DataFrames ---
        data_frames = self._prepare_data_frames(data)

        # Determine which variables to plot
        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        # --- Iterate over datasets ---
        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = axes.flatten() if num_vars > 1 else [axes]

            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]

                # Perform automatic trimming
                ds = DataStream(df)
                trimmed_ds = ds.trim(
                    column_name=column,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )

                # --- If steady state detected ---
                if trimmed_ds is not None and not trimmed_ds.df.empty:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        steady_state_start,
                        color="r",
                        linestyle="--",
                        label="Steady State Start",
                    )
                    ax.axhline(
                        overall_mean,
                        color="g",
                        linestyle="-",
                        label="Mean (Post-SS)",
                    )

                    # Conditional shading: always for 'std', optional for others
                    if method == "std" or show_std_bands:
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - overall_std,
                            overall_mean + overall_std,
                            color="blue",
                            alpha=0.3,
                            label="±1 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 2 * overall_std,
                            overall_mean + 2 * overall_std,
                            color="yellow",
                            alpha=0.2,
                            label="±2 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 3 * overall_std,
                            overall_mean + 3 * overall_std,
                            color="red",
                            alpha=0.1,
                            label="±3 Std Dev",
                        )

                    # --- Statistical annotation box ---
                    stats = trimmed_ds.compute_statistics(
                        column_name=column
                    )["results"][column]
                    ci_low, ci_high = stats["confidence_interval"]
                    pm_low, pm_high = stats["pm_std"]
                    txt = (
                        f"μ = {stats['mean']:.3f}\n"
                        f"SEM = {stats['mean_uncertainty']:.3f}\n"
                        f"CI = [{ci_low:.3f}, {ci_high:.3f}]\n"
                        f"±STD = [{pm_low:.3f}, {pm_high:.3f}]"
                    )
                    ax.text(
                        0.03,
                        0.97,
                        txt,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize="small",
                        bbox=dict(
                            facecolor="white", alpha=0.6, edgecolor="gray"
                        ),
                    )

                # --- If no steady state detected ---
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"[⚠] {column}: no steady-state detected or insufficient data."
                    )

                # --- Axis formatting ---
                ax.set_title(
                    f"Steady-State Detection with Stats (Flex): {column}"
                )
                ax.set_xlabel("Time")
                ax.set_ylabel(f"{column} (Values)")
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            # Remove unused axes
            for k in range(len(variables_to_plot), len(axes)):
                fig.delaxes(axes[k])

            # Layout + save option
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_with_stats_flex_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=300)
                print(
                    f"✅ Saved flexible annotated steady-state plot to {save_path}"
                )
            plt.show()
            plt.close(fig)

    def steady_state_automatic_plot(
        self,
        data,
        variables_to_plot=None,
        batch_size=10,
        start_time=0.0,
        method="std",
        threshold=None,
        robust=True,
        show_std_bands=False,
        save=False,
        y_range=None,
    ):
        """
        Modified steady-state detection plot:
        - Larger font sizes
        - Y-axis labeled "Heat Flux"
        - Green mean line starts only after steady-state start
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [
                col for col in first_df.columns if col != "time"
            ]
        else:
            variables_to_plot = [
                var for var in variables_to_plot if var != "time"
            ]

        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig_size = self._calc_fig_size(num_cols, num_rows)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axes = [axes] if num_vars == 1 else axes.flatten()

            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]

                ds = DataStream(df)
                trimmed_ds = ds.trim(
                    column,
                    batch_size=batch_size,
                    start_time=start_time,
                    method=method,
                    threshold=threshold,
                    robust=robust,
                )

                if trimmed_ds is not None and not trimmed_ds.df.empty:
                    steady_state_start = trimmed_ds.df["time"].iloc[0]
                    after_ss = signal[time >= steady_state_start]
                    overall_mean = after_ss.mean()
                    overall_std = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=steady_state_start,
                        color="r",
                        linestyle="--",
                        linewidth=2,
                        label="Steady-State Start",
                    )

                    # Only draw mean line starting from steady-state region
                    ax.plot(
                        time[time >= steady_state_start],
                        np.ones(len(time[time >= steady_state_start]))
                        * overall_mean,
                        color="g",
                        linestyle="-",
                        linewidth=2.5,
                        label="Mean (Post-SS)",
                    )

                    # Optional shaded std bands
                    if method == "std" or show_std_bands:
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - overall_std,
                            overall_mean + overall_std,
                            color="blue",
                            alpha=0.3,
                            label="±1 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 2 * overall_std,
                            overall_mean + 2 * overall_std,
                            color="yellow",
                            alpha=0.2,
                            label="±2 Std Dev",
                        )
                        ax.fill_between(
                            time[time >= steady_state_start],
                            overall_mean - 3 * overall_std,
                            overall_mean + 3 * overall_std,
                            color="red",
                            alpha=0.1,
                            label="±3 Std Dev",
                        )
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(f"[⚠] {column}: steady state not detected.")

                # ax.set_title(
                #    f"Steady-State Detection of {column}",
                #    fontsize=18,
                #    fontweight="bold",
                # )
                ax.set_xlabel("Time", fontsize=16, fontweight="bold")
                ax.set_ylabel("Heat Flux", fontsize=16, fontweight="bold")
                ax.tick_params(axis="both", labelsize=14)
                ax.legend(fontsize=13)
                ax.grid(True, alpha=0.3)

                if y_range is not None:
                    ax.set_ylim(y_range)

            for k in range(len(variables_to_plot), len(axes)):
                fig.delaxes(axes[k])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                save_path = os.path.join(
                    self.output_dir,
                    f"steady_state_auto_custom_{self.format_dataset_name(dataset_name)}.png",
                )
                plt.savefig(save_path, dpi=350, bbox_inches="tight")
                print(f"✅ Saved plot to {save_path}")
            plt.show()
            plt.close(fig)

    def plot_ensemble_gg(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
        condensed_legend=False,
        y_range=None,
    ):
        """
        Modified ensemble plot:
        - Larger font sizes for all text
        - Y-axis labeled "Heat Flux"
        """
        member_dfs = {
            f"Member {i}": ds.df
            for i, ds in enumerate(ensemble_obj.data_streams)
        }
        avg_stream = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_stream.df

        all_cols = avg_stream.df.columns.tolist()
        vars_to_plot = (
            [c for c in all_cols if c != "time"]
            if variables_to_plot is None
            else [c for c in variables_to_plot if c != "time"]
        )
        if not vars_to_plot:
            raise ValueError("No variables to plot.")
        n_vars = len(vars_to_plot)

        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            first_drawn = False
            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"],
                        df[var],
                        label="Ensemble Average",
                        color="black",
                        linewidth=3,
                        zorder=5,
                    )
                else:
                    label = (
                        "Individual Members"
                        if condensed_legend and not first_drawn
                        else None
                    )
                    ax.plot(
                        df["time"],
                        df[var],
                        label=label if not condensed_legend else label,
                        alpha=0.35,
                        linewidth=1.2,
                    )
                    first_drawn = True

            ax.set_title(
                f"Ensemble Mean and Members of {var}",
                fontsize=18,
                fontweight="bold",
            )
            ax.set_xlabel("Time", fontsize=16, fontweight="bold")
            ax.set_ylabel("Heat Flux", fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.3)

            if y_range is not None:
                ax.set_ylim(y_range)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        if condensed_legend:
            unique = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen and l is not None:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique)

        legend_ncol = min(len(labels), 3)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.0, -0.08),
            ncol=legend_ncol,
            fontsize=13,
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_heatflux_plot.png"
            )
            fig.savefig(outpath, dpi=350, bbox_inches="tight")
            print(f"✅ Saved ensemble plot to {outpath}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_ensemble__(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
        condensed_legend=False,
        y_range=None,
    ):
        """
        Modified ensemble plot:
        - Larger font sizes for all text
        - Y-axis labeled "Heat Flux"
        - Bold, larger legend for poster clarity
        """
        member_dfs = {
            f"Member {i}": ds.df
            for i, ds in enumerate(ensemble_obj.data_streams)
        }
        avg_stream = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_stream.df

        all_cols = avg_stream.df.columns.tolist()
        vars_to_plot = (
            [c for c in all_cols if c != "time"]
            if variables_to_plot is None
            else [c for c in variables_to_plot if c != "time"]
        )
        if not vars_to_plot:
            raise ValueError("No variables to plot.")
        n_vars = len(vars_to_plot)

        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            first_drawn = False
            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"],
                        df[var],
                        label="Ensemble Average",
                        color="black",
                        linewidth=3,
                        zorder=5,
                    )
                else:
                    label = (
                        "Individual Members"
                        if condensed_legend and not first_drawn
                        else None
                    )
                    ax.plot(
                        df["time"],
                        df[var],
                        label=label if not condensed_legend else label,
                        alpha=0.35,
                        linewidth=1.2,
                    )
                    first_drawn = True

            # ax.set_title(
            #    f"Ensemble Mean and Members of {var}",
            #    fontsize=18,
            #    fontweight="bold",
            # )
            ax.set_xlabel("Time", fontsize=16, fontweight="bold")
            ax.set_ylabel("Heat Flux", fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.3)

            if y_range is not None:
                ax.set_ylim(y_range)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        if condensed_legend:
            unique = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen and l is not None:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique)

        legend_ncol = min(len(labels), 3)
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.2, -0.18),
            ncol=legend_ncol,
            fontsize=15,  # Larger text
            prop={"weight": "bold"},  # Bold legend font
            frameon=True,  # Adds background box
            facecolor="white",  # White background
            framealpha=0.85,  # Semi-transparent for clarity
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_heatflux_plot.png"
            )
            fig.savefig(outpath, dpi=350, bbox_inches="tight")
            print(f"✅ Saved ensemble plot to {outpath}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_ensemble(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
        condensed_legend=False,
        y_range=None,
    ):
        """
        Modified ensemble plot:
        - Larger font sizes for all text
        - Y-axis labeled "Heat Flux"
        - Bold, larger legend for poster clarity
        - Legend placement:
            * Inside plot (upper right) if only one variable
            * Bottom-center if multiple variables
        """
        member_dfs = {
            f"Member {i}": ds.df
            for i, ds in enumerate(ensemble_obj.data_streams)
        }
        avg_stream = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_stream.df

        all_cols = avg_stream.df.columns.tolist()
        vars_to_plot = (
            [c for c in all_cols if c != "time"]
            if variables_to_plot is None
            else [c for c in variables_to_plot if c != "time"]
        )
        if not vars_to_plot:
            raise ValueError("No variables to plot.")
        n_vars = len(vars_to_plot)

        ncols = 2
        nrows = math.ceil(n_vars / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False
        )
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            first_drawn = False
            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"],
                        df[var],
                        label="Ensemble Average",
                        color="black",
                        linewidth=3,
                        zorder=5,
                    )
                else:
                    label = (
                        "Individual Members"
                        if condensed_legend and not first_drawn
                        else None
                    )
                    ax.plot(
                        df["time"],
                        df[var],
                        label=label if not condensed_legend else label,
                        alpha=0.35,
                        linewidth=1.2,
                    )
                    first_drawn = True

            ax.set_xlabel("Time", fontsize=16, fontweight="bold")
            ax.set_ylabel("Heat Flux", fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", labelsize=14)
            ax.grid(True, alpha=0.3)

            if y_range is not None:
                ax.set_ylim(y_range)

        # Remove unused subplot slots
        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        # Collect legend handles and labels
        handles, labels = axes[0].get_legend_handles_labels()
        if condensed_legend:
            unique = []
            seen = set()
            for h, l in zip(handles, labels):
                if l not in seen and l is not None:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique)

        legend_ncol = min(len(labels), 3)

        # === Legend placement logic ===
        if n_vars == 1:
            # Single variable → Legend inside plot (upper right)
            ax = axes[0]
            legend = ax.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=15,
                prop={"weight": "bold"},
                frameon=True,
                facecolor="white",
                framealpha=0.85,
            )
        else:
            # Multiple variables → Legend at bottom center
            legend = fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=legend_ncol,
                fontsize=15,
                prop={"weight": "bold"},
                frameon=True,
                facecolor="white",
                framealpha=0.85,
            )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_heatflux_plot.png"
            )
            fig.savefig(outpath, dpi=350, bbox_inches="tight")
            print(f"✅ Saved ensemble plot to {outpath}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)
