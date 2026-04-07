import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

from quends.base.data_stream import DataStream
from quends.base.ensemble import Ensemble


class Plotter:
    """
    Plotting utilities for DataStream and Ensemble objects.

    All methods accept DataStream instances, Ensemble instances, or plain
    dicts of DataFrames.  The plotter works with the NEW DataStream API
    (.data property, not the legacy .df attribute).

    Usage
    -----
    plotter = Plotter(output_dir="figures")
    plotter.trace_plot(ds, variables_to_plot=["phi2"], save=True)
    """

    def __init__(self, output_dir="results_figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_dataset_name(dataset_name):
        """Return a title-cased, space-separated version of a dataset name."""
        return dataset_name.replace("_", " ").title()

    def _prepare_data_frames(self, data):
        """
        Return a ``{name: DataFrame}`` dict from *data*.

        Accepted input types
        --------------------
        DataStream  ->  {"DataStream": ds.data}
        Ensemble    ->  {"Member 0": ds.data, "Member 1": ds.data, ...}
        dict        ->  passed through unchanged
        """
        if isinstance(data, DataStream):
            return {"DataStream": data.data}
        elif isinstance(data, Ensemble):
            return {
                f"Member {k}": ds.data
                for k, ds in enumerate(data.data_streams)
            }
        elif isinstance(data, dict):
            return data
        else:
            raise ValueError(
                "data must be a DataStream, Ensemble, or dict of DataFrames."
            )

    def _calc_fig_size(self, num_cols, num_rows):
        """Return (width, height) scaled to the subplot grid."""
        return (max(8, num_cols * 3), max(6, num_rows * 3))

    @staticmethod
    def _trim_datastream(ds, column, method="std", batch_size=10,
                         start_time=0.0, threshold=None, robust=True):
        """
        Trim *ds* on *column* via the unified :mod:`quends.base.trim` strategy system.

        Adapter: delegates to :func:`~quends.base.trim.build_trim_strategy` and
        :class:`~quends.base.trim.TrimDataStreamOperation` — the canonical low-level
        trim path in ``trim.py``.

        Parameters
        ----------
        ds : DataStream
        column : str
        method : str
            ``"std"`` | ``"threshold"`` | ``"rolling_variance"`` |
            ``"self_consistent"`` | ``"iqr"``
        batch_size : int
            Passed as ``window_size`` to the strategy.
        start_time : float
        threshold : float or None
        robust : bool

        Returns
        -------
        DataStream
            Trimmed DataStream (may be empty if no steady state detected).
        """
        from quends.base.trim import build_trim_strategy, TrimDataStreamOperation

        strategy = build_trim_strategy(
            method=method,
            window_size=batch_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )
        op = TrimDataStreamOperation(strategy=strategy)
        return op(ds, column_name=column)

    # ------------------------------------------------------------------ #
    #  Single-DataStream trace plots                                        #
    # ------------------------------------------------------------------ #

    def trace_plot(self, data, variables_to_plot=None, save=False):
        """
        Plot raw time-series traces for each variable.

        Parameters
        ----------
        data : DataStream, Ensemble, or dict
        variables_to_plot : list of str, optional
        save : bool
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [c for c in first_df.columns if c != "time"]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for dataset_name, df in data_frames.items():
            time_series = df["time"]
            num_traces = len(variables_to_plot)
            num_cols = min(5, num_traces)
            num_rows = (num_traces + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=self._calc_fig_size(num_cols, num_rows)
            )
            axes = [axes] if num_traces == 1 else axes.flatten()

            for j, column in enumerate(variables_to_plot):
                axes[j].plot(time_series, df[column], label=column)
                axes[j].set_xlabel("Time")
                axes[j].set_ylabel(column)
                axes[j].set_title(column)
                axes[j].legend(fontsize="small")
                axes[j].grid(True)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Time Series — {self.format_dataset_name(dataset_name)}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                plt.savefig(
                    os.path.join(
                        self.output_dir,
                        f"time_series_{self.format_dataset_name(dataset_name)}.png",
                    ),
                    dpi=150,
                )
            plt.show()
            plt.close(fig)

        return axes

    def trace_plot_with_mean(self, data, variables_to_plot=None,
                             window_size=None, save=False):
        """
        Plot each trace with the block-mean and 95 % CI overlaid.

        Calls ``DataStream.compute_statistics()`` internally — no trimming
        is applied.  Pass a pre-trimmed DataStream if you want the mean
        computed only on the steady-state portion.

        Parameters
        ----------
        data : DataStream, Ensemble, or dict
        variables_to_plot : list of str, optional
        window_size : int or None
            Passed to ``compute_statistics()``.  None = auto-tune.
        save : bool
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [c for c in first_df.columns if c != "time"]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for dataset_name, df in data_frames.items():
            num_traces = len(variables_to_plot)
            num_cols = min(5, num_traces)
            num_rows = (num_traces + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=self._calc_fig_size(num_cols, num_rows)
            )
            axes = [axes] if num_traces == 1 else axes.flatten()

            ds = DataStream(df)

            for j, column in enumerate(variables_to_plot):
                stats = ds.compute_statistics(
                    column_name=column, window_size=window_size
                )
                s = stats.get(column, {})
                mu = s.get("mean")
                ci = s.get("confidence_interval", (None, None))

                axes[j].plot(df["time"], df[column], label=column, alpha=0.7)
                if mu is not None:
                    axes[j].axhline(
                        y=mu, color="red", linestyle="-", label=f"Mean={mu:.4g}"
                    )
                if ci[0] is not None and ci[1] is not None:
                    axes[j].fill_between(
                        df["time"], ci[0], ci[1],
                        color="red", alpha=0.15, label="95 % CI",
                    )
                axes[j].set_xlabel("Time")
                axes[j].set_ylabel(column)
                axes[j].set_title(column)
                axes[j].legend(fontsize="small")
                axes[j].grid(True)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Trace with Mean — {self.format_dataset_name(dataset_name)}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                plt.savefig(
                    os.path.join(
                        self.output_dir,
                        f"trace_with_mean_{self.format_dataset_name(dataset_name)}.png",
                    ),
                    dpi=150,
                )
            plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Ensemble trace plots                                                 #
    # ------------------------------------------------------------------ #

    def ensemble_trace_plot(self, data, variables_to_plot=None, save=False):
        """
        Overlay traces from all ensemble members on one plot per variable.

        Parameters
        ----------
        data : DataStream, Ensemble, or dict
        variables_to_plot : list of str, optional
        save : bool
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [c for c in first_df.columns if c != "time"]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for var in variables_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                ax.plot(df["time"], df[var], label=dataset_name)
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            ax.set_title(f"Ensemble Time Series — {var}")
            ax.legend(fontsize="small")
            ax.grid(True)
            plt.tight_layout()
            if save:
                plt.savefig(
                    os.path.join(self.output_dir, f"ensemble_trace_{var}.png"),
                    dpi=150,
                )
            plt.show()
            plt.close(fig)

    def ensemble_trace_plot_with_mean(self, data, variables_to_plot=None,
                                      window_size=None, save=False):
        """
        Overlay ensemble member traces with the per-member block mean.

        Parameters
        ----------
        data : DataStream, Ensemble, or dict
        variables_to_plot : list of str, optional
        window_size : int or None
        save : bool
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [c for c in first_df.columns if c != "time"]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for var in variables_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                ds = DataStream(df)
                stats = ds.compute_statistics(
                    column_name=var, window_size=window_size
                )
                mu = stats.get(var, {}).get("mean")
                ax.plot(df["time"], df[var], label=dataset_name, alpha=0.6)
                if mu is not None:
                    ax.axhline(
                        y=mu, linestyle="--", linewidth=1.0,
                        label=f"{dataset_name} mean={mu:.4g}",
                    )
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            ax.set_title(f"Ensemble Traces with Means — {var}")
            ax.legend(fontsize="small")
            ax.grid(True)
            plt.tight_layout()
            if save:
                plt.savefig(
                    os.path.join(
                        self.output_dir, f"ensemble_trace_with_mean_{var}.png"
                    ),
                    dpi=150,
                )
            plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Steady-state detection plots (single DataStream)                    #
    # ------------------------------------------------------------------ #

    def steady_state_automatic_plot(
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
        Auto-detect steady-state start for each variable and annotate the plot.

        Uses the NEW strategy-based trim system (``_trim_datastream``).
        If a steady state is found, the plot shows the full signal with:
          - vertical dashed red line at SS start
          - horizontal green mean line
          - ±1/±2/±3 std-dev shaded bands

        Parameters
        ----------
        data : DataStream, Ensemble, or dict
        variables_to_plot : list of str, optional
        batch_size : int
            Window size passed to the trim strategy.
        start_time : float
        method : str
            'std', 'threshold', 'rolling_variance', 'self_consistent', 'iqr'
        threshold : float or None
        robust : bool
        save : bool
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [c for c in first_df.columns if c != "time"]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for dataset_name, df in data_frames.items():
            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=self._calc_fig_size(num_cols, num_rows)
            )
            axes = [axes] if num_vars == 1 else axes.flatten()

            for idx, column in enumerate(variables_to_plot):
                ax = axes[idx]
                time = df["time"]
                signal = df[column]

                ds = DataStream(df)
                trimmed_ds = self._trim_datastream(
                    ds, column,
                    method=method, batch_size=batch_size,
                    start_time=start_time, threshold=threshold, robust=robust,
                )

                if trimmed_ds is not None and not trimmed_ds.data.empty:
                    ss_start = trimmed_ds.data["time"].iloc[0]
                    after_ss = signal[time >= ss_start]
                    mu = after_ss.mean()
                    sigma = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=ss_start, color="r", linestyle="--",
                        label="SS Start",
                    )
                    ax.axhline(y=mu, color="g", linestyle="-", label="Mean")
                    ax.fill_between(
                        time[time >= ss_start],
                        mu - sigma, mu + sigma,
                        color="blue", alpha=0.3, label="±1 Std",
                    )
                    ax.fill_between(
                        time[time >= ss_start],
                        mu - 2 * sigma, mu + 2 * sigma,
                        color="yellow", alpha=0.2, label="±2 Std",
                    )
                    ax.fill_between(
                        time[time >= ss_start],
                        mu - 3 * sigma, mu + 3 * sigma,
                        color="red", alpha=0.1, label="±3 Std",
                    )
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(f"{column}: no steady state detected — plotting full signal.")

                ax.set_title(column)
                ax.set_xlabel("Time")
                ax.set_ylabel(column)
                ax.legend(fontsize="small")
                ax.grid(True)

            for k in range(idx + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Steady-State Detection — {self.format_dataset_name(dataset_name)}",
                fontsize=14,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                plt.savefig(
                    os.path.join(
                        self.output_dir,
                        f"steady_state_auto_{self.format_dataset_name(dataset_name)}.png",
                    ),
                    dpi=150,
                )
            plt.show()
            plt.close(fig)

    def steady_state_plot(
        self, data, variables_to_plot=None, steady_state_start=None, save=False
    ):
        """
        Plot steady-state annotation using a user-supplied start time.

        *steady_state_start* may be a single float (applied to all variables)
        or a dict mapping variable name → float.

        Parameters
        ----------
        data : DataStream, Ensemble, or dict
        variables_to_plot : list of str, optional
        steady_state_start : float or dict, optional
        save : bool
        """
        data_frames = self._prepare_data_frames(data)

        if variables_to_plot is None:
            first_df = next(iter(data_frames.values()))
            variables_to_plot = [c for c in first_df.columns if c != "time"]
        else:
            variables_to_plot = [v for v in variables_to_plot if v != "time"]

        for dataset_name, df in data_frames.items():
            if "time" not in df.columns:
                raise ValueError(
                    f"DataFrame for '{dataset_name}' is missing a 'time' column."
                )

            num_vars = len(variables_to_plot)
            num_cols = min(5, num_vars)
            num_rows = (num_vars + num_cols - 1) // num_cols
            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=self._calc_fig_size(num_cols, num_rows)
            )
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
                    mu = after_ss.mean()
                    sigma = after_ss.std()

                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(
                        x=manual_ss, color="r", linestyle="--",
                        label="SS Start",
                    )
                    ax.axhline(y=mu, color="g", linestyle="-", label="Mean")
                    ax.fill_between(
                        time[time >= manual_ss],
                        mu - sigma, mu + sigma,
                        color="blue", alpha=0.3, label="±1 Std",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        mu - 2 * sigma, mu + 2 * sigma,
                        color="yellow", alpha=0.2, label="±2 Std",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        mu - 3 * sigma, mu + 3 * sigma,
                        color="red", alpha=0.1, label="±3 Std",
                    )
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    print(
                        f"{column}: no steady state start provided — plotting raw signal."
                    )

                ax.set_title(column)
                ax.set_xlabel("Time")
                ax.set_ylabel(column)
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Steady-State (Manual) — {self.format_dataset_name(dataset_name)}",
                fontsize=14,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            if save:
                plt.savefig(
                    os.path.join(
                        self.output_dir,
                        f"steady_state_manual_{self.format_dataset_name(dataset_name)}.png",
                    ),
                    dpi=150,
                )
            plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------ #
    #  ACF plots                                                            #
    # ------------------------------------------------------------------ #

    def plot_acf(self, data, alpha=0.05, column=None, ax=None):
        """
        Plot the Autocorrelation Function (ACF).

        Parameters
        ----------
        data : DataStream or array-like
            If DataStream, the *column* column is used.
        alpha : float
            Significance level for the confidence band.
        column : str, optional
            Column to use when *data* is a DataStream.
        ax : matplotlib.axes.Axes, optional
            Plot into this axis; create a new figure if None.
        """
        if isinstance(data, DataStream):
            df = data.data
            if column is None:
                cols = [c for c in df.columns if c != "time"]
                if not cols:
                    raise ValueError("No valid column found in the DataStream.")
                column = cols[0]
            filtered = df[column].dropna().values
        else:
            filtered = np.array(data).flatten()
            if column is None:
                column = "signal"

        n = len(filtered)
        if n == 0:
            raise ValueError("Data is empty after filtering.")

        nlags = max(1, int(n / 3))
        acf_values = acf(filtered, nlags=nlags, fft=False)
        conf = norm.ppf(1 - alpha / 2) / np.sqrt(n)

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.stem(range(len(acf_values)), acf_values, basefmt=" ")
        ax.axhline(conf, color="red", linestyle="--",
                   label=f"95 % CI upper: {conf:.3f}")
        ax.axhline(-conf, color="red", linestyle="--",
                   label=f"95 % CI lower: {-conf:.3f}")
        ax.set_title(f"ACF — '{column}'")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        if own_fig:
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def plot_acf_ensemble(self, ensemble_obj, alpha=0.05, column=None,
                          save=False):
        """
        ACF grid — one subplot per ensemble member.

        Parameters
        ----------
        ensemble_obj : Ensemble
        alpha : float
        column : str, optional
        save : bool
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            self.plot_acf(ds, alpha=alpha, column=column, ax=axes[i])
            axes[i].set_title(f"Member {i} — ACF")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save:
            plt.savefig(
                os.path.join(self.output_dir, "ensemble_acf.png"), dpi=150
            )
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Ensemble steady-state plots                                          #
    # ------------------------------------------------------------------ #

    def ensemble_steady_state_automatic_plot(
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
        Auto-detect steady state for each member and display one subplot per member.

        Uses the NEW strategy-based trim system (``_trim_datastream``).

        Parameters
        ----------
        ensemble_obj : Ensemble
        variables_to_plot : list of str, optional
        batch_size : int
        start_time : float
        method : str
        threshold : float or None
        robust : bool
        save : bool
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(math.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.data
            if variables_to_plot is None:
                vars_plot = [c for c in df.columns if c != "time"]
            else:
                vars_plot = [v for v in variables_to_plot if v != "time"]

            ax = axes[i]
            time = df["time"]

            for var in vars_plot:
                signal = df[var]
                trimmed = self._trim_datastream(
                    ds, var,
                    method=method, batch_size=batch_size,
                    start_time=start_time, threshold=threshold, robust=robust,
                )
                if trimmed is not None and not trimmed.data.empty:
                    ss_start = trimmed.data["time"].iloc[0]
                    after_ss = signal[time >= ss_start]
                    mu = after_ss.mean()
                    sigma = after_ss.std()

                    ax.plot(time, signal, label=var, alpha=0.7)
                    ax.axvline(x=ss_start, color="r", linestyle="--",
                               label="SS Start")
                    ax.axhline(y=mu, color="g", linestyle="-", label="Mean")
                    ax.fill_between(
                        time[time >= ss_start],
                        mu - sigma, mu + sigma,
                        color="blue", alpha=0.3, label="±1 Std",
                    )
                    ax.fill_between(
                        time[time >= ss_start],
                        mu - 2 * sigma, mu + 2 * sigma,
                        color="yellow", alpha=0.2, label="±2 Std",
                    )
                    ax.fill_between(
                        time[time >= ss_start],
                        mu - 3 * sigma, mu + 3 * sigma,
                        color="red", alpha=0.1, label="±3 Std",
                    )
                else:
                    ax.plot(time, signal, label=var, alpha=0.7)
                    print(f"Member {i} / {var}: no SS detected.")

            ax.set_title(f"Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(", ".join(vars_plot))
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(
                self.output_dir, "ensemble_steady_state_auto.png"
            )
            plt.savefig(save_path, dpi=150)
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
        Annotate each ensemble member with a user-supplied SS start time.

        *steady_state_start* may be a float or a ``{var_name: float}`` dict.

        Parameters
        ----------
        ensemble_obj : Ensemble
        variables_to_plot : list of str, optional
        steady_state_start : float or dict, optional
        save : bool
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.data
            if variables_to_plot is None:
                vars_plot = [c for c in df.columns if c != "time"]
            else:
                vars_plot = [v for v in variables_to_plot if v != "time"]

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
                    mu = after_ss.mean()
                    sigma = after_ss.std()

                    ax.plot(time, signal, label=var, alpha=0.7)
                    ax.axvline(x=manual_ss, color="r", linestyle="--",
                               label="SS Start")
                    ax.axhline(y=mu, color="g", linestyle="-", label="Mean")
                    ax.fill_between(
                        time[time >= manual_ss],
                        mu - sigma, mu + sigma,
                        color="blue", alpha=0.3, label="±1 Std",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        mu - 2 * sigma, mu + 2 * sigma,
                        color="yellow", alpha=0.2, label="±2 Std",
                    )
                    ax.fill_between(
                        time[time >= manual_ss],
                        mu - 3 * sigma, mu + 3 * sigma,
                        color="red", alpha=0.1, label="±3 Std",
                    )
                else:
                    ax.plot(time, signal, label=var, alpha=0.7)
                    print(f"Member {i} / {var}: no SS start provided.")

            ax.set_title(f"Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(", ".join(vars_plot))
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            save_path = os.path.join(
                self.output_dir, "ensemble_steady_state_manual.png"
            )
            plt.savefig(save_path, dpi=150)
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Ensemble member + average plots                                      #
    # ------------------------------------------------------------------ #

    def plot_ensemble(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
    ):
        """
        Plot individual ensemble members and their ensemble average.

        One subplot per variable, 2-column grid.  The ensemble average is
        drawn as a thick black line; members are drawn as thin translucent
        lines.

        Parameters
        ----------
        ensemble_obj : Ensemble
        variables_to_plot : list of str, optional
        show_plots : bool
        save : bool
        """
        member_dfs = {
            f"Member {i}": ds.data
            for i, ds in enumerate(ensemble_obj.data_streams)
        }
        avg_ds = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_ds.data

        all_cols = avg_ds.data.columns.tolist()
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
                        df["time"], df[var],
                        label="Ensemble Average",
                        color="black", linewidth=2.5, zorder=5,
                    )
                else:
                    ax.plot(
                        df["time"], df[var],
                        label=name, alpha=0.3, linewidth=1.0,
                    )
            ax.set_title(var, fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(var, fontsize=12)
            ax.grid(True, alpha=0.3)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        legend_ncol = min(len(labels), 4)
        legend_nrow = math.ceil(len(labels) / legend_ncol)
        legend_y = -0.05 * legend_nrow
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_ncol, fontsize="small", frameon=False,
        )

        bottom_margin = 0.005 * legend_nrow + 0.02
        plt.tight_layout(rect=[0, bottom_margin, 1, 1])

        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_members_and_average.png"
            )
            fig.savefig(outpath, dpi=150)
            print(f"Figure saved to {outpath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_ensemble_with_average(
        self,
        ensemble_obj,
        variables_to_plot=None,
        show_plots=False,
        save=False,
        condensed_legend=False,
        y_range=None,
    ):
        """
        Enhanced ensemble plot with optional condensed legend and y-axis range.

        Compared to ``plot_ensemble``, this method adds:
          - ``condensed_legend=True``: all member traces share one legend entry
            ("Individual Members") while the average retains its label.
          - ``y_range``: explicit ``(y_min, y_max)`` applied to every subplot.

        Parameters
        ----------
        ensemble_obj : Ensemble
        variables_to_plot : list of str, optional
        show_plots : bool
        save : bool
        condensed_legend : bool
            If True, group all member traces under one legend entry.
        y_range : tuple (ymin, ymax) or None
            If provided, applies ``ax.set_ylim(y_range)`` to each subplot.
        """
        member_dfs = {
            f"Member {i}": ds.data
            for i, ds in enumerate(ensemble_obj.data_streams)
        }
        avg_ds = ensemble_obj.compute_average_ensemble()
        member_dfs["Ensemble Average"] = avg_ds.data

        all_cols = avg_ds.data.columns.tolist()
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
            first_drawn = False

            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(
                        df["time"], df[var],
                        label="Ensemble Average",
                        color="black", linewidth=2.5, zorder=5,
                    )
                else:
                    if condensed_legend:
                        label = "Individual Members" if not first_drawn else None
                    else:
                        label = name
                    ax.plot(
                        df["time"], df[var],
                        label=label, alpha=0.3, linewidth=1.0,
                    )
                    first_drawn = True

            ax.set_title(var, fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(var, fontsize=12)
            ax.grid(True, alpha=0.3)
            if y_range is not None:
                ax.set_ylim(y_range)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        # Build legend (de-duplicate if condensed)
        handles, labels = axes[0].get_legend_handles_labels()
        if condensed_legend:
            seen, unique = set(), []
            for h, l in zip(handles, labels):
                if l is not None and l not in seen:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique) if unique else ([], [])

        legend_ncol = min(len(labels), 4)
        legend_nrow = max(1, math.ceil(len(labels) / legend_ncol))
        legend_y = -0.05 if n_vars == 1 else -0.08
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=legend_ncol, fontsize="small", frameon=False,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save:
            outpath = os.path.join(
                self.output_dir, "ensemble_with_average.png"
            )
            fig.savefig(outpath, dpi=150)
            print(f"Figure saved to {outpath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
