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
            num_rows = (
                num_traces + num_cols - 1
            ) // num_cols  # Ceiling division
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
            num_rows = (
                num_traces + num_cols - 1
            ) // num_cols  # Ceiling division
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
        import math

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
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        plt.show()

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
            ax.set_title(var, fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(var, fontsize=12)
            ax.grid(True)

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
            fig.savefig(outpath, dpi=150)
        if show_plots:
            plt.show()
        else:
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

                    # Compute and annotate stats
                    stats = trimmed_ds.compute_statistics(column_name=column)[
                        column
                    ]
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
                    ax.set_title(column)
                    ax.set_xlabel("Time")
                    ax.set_ylabel(column)
                    ax.legend(fontsize="small")
                    ax.grid(True)
                    print(
                        f"{column}: No steady state detected or insufficient data."
                    )

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
