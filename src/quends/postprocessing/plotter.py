import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

from quends.base.data_stream import DataStream
from quends.base.ensemble import Ensemble

logger = logging.getLogger(__name__)


class Plotter:
    """
    Plotting utilities for DataStream and Ensemble objects.

    Uniform plotting contract
    --------------------------
    Every public plotting method accepts the same output-control keywords and
    returns the Matplotlib objects instead of forcing display:

    * ``save`` (bool)        — write the figure to disk.
    * ``show`` (bool)        — call ``plt.show()`` (default **False**; scripts/CI safe).
    * ``filename`` (str)     — output filename (single-figure methods).
    * ``output_dir`` (str)   — per-call output directory (defaults to the ctor dir).
    * ``overwrite`` (bool)   — allow clobbering an existing file (default False).
    * ``dpi`` (int)          — raster resolution when saving.

    Return value:

    * single-figure methods return ``(fig, axes)``;
    * multi-figure methods (one figure per variable/member) return a list of
      ``(fig, axes)`` tuples.

    Regenerate-from-CSV
    -------------------
    The stats/trim-heavy methods accept optional *precomputed* arguments
    (``stats=``, ``ss_starts=``, ``means=``, ``avg_df=``, ``acf_values=``). When
    supplied, the method skips the expensive computation and only renders — so a
    figure can be rebuilt from saved results without recomputing statistics.
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
        """Return a ``{name: DataFrame}`` dict from a DataStream/Ensemble/dict."""
        if isinstance(data, DataStream):
            return {"DataStream": data.data}
        elif isinstance(data, Ensemble):
            return {f"Member {k}": ds.data for k, ds in enumerate(data.data_streams)}
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
    def _select_vars(first_df, variables_to_plot):
        """Resolve the list of variables to plot (always excluding 'time')."""
        if variables_to_plot is None:
            return [c for c in first_df.columns if c != "time"]
        return [v for v in variables_to_plot if v != "time"]

    @staticmethod
    def _draw_std_bands(ax, x, mu, sigma):
        """Draw shared ±1/±2/±3 standard-deviation bands on *ax*."""
        ax.fill_between(x, mu - sigma, mu + sigma, color="blue", alpha=0.3, label="±1 Std")
        ax.fill_between(x, mu - 2 * sigma, mu + 2 * sigma, color="yellow", alpha=0.2, label="±2 Std")
        ax.fill_between(x, mu - 3 * sigma, mu + 3 * sigma, color="red", alpha=0.1, label="±3 Std")

    def _finalize(self, fig, axes, *, default_name, save=False, show=False,
                  filename=None, output_dir=None, overwrite=False, dpi=150):
        """Save (with overwrite guard), optionally show, always return (fig, axes)."""
        if save:
            out_dir = output_dir or self.output_dir
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename or default_name)
            if os.path.exists(path) and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing figure: {path}. "
                    f"Pass overwrite=True to allow."
                )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info("Figure saved to %s", path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    @staticmethod
    def _trim_datastream(ds, column, method="std", batch_size=10,
                         start_time=0.0, threshold=None, robust=True):
        """Trim *ds* on *column* via the unified trim strategy system."""
        from quends.base.trim import build_trim_strategy, TrimDataStreamOperation

        strategy = build_trim_strategy(
            method=method, window_size=batch_size, start_time=start_time,
            threshold=threshold, robust=robust,
        )
        op = TrimDataStreamOperation(strategy=strategy)
        return op(ds, column_name=column)

    # ------------------------------------------------------------------ #
    #  Single-DataStream trace plots                                        #
    # ------------------------------------------------------------------ #

    def trace_plot(self, data, variables_to_plot=None, *, save=False, show=False,
                   filename=None, output_dir=None, overwrite=False, dpi=150):
        """Plot raw time-series traces. Returns a list of (fig, axes)."""
        data_frames = self._prepare_data_frames(data)
        variables_to_plot = self._select_vars(
            next(iter(data_frames.values())), variables_to_plot
        )

        results = []
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
                f"Time Series — {self.format_dataset_name(dataset_name)}", fontsize=16
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            results.append(self._finalize(
                fig, axes,
                default_name=f"time_series_{self.format_dataset_name(dataset_name)}.png",
                save=save, show=show, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            ))
        return results

    def trace_plot_with_mean(self, data, variables_to_plot=None, window_size=None,
                             *, stats=None, save=False, show=False, filename=None,
                             output_dir=None, overwrite=False, dpi=150):
        """Plot each trace with block-mean + 95% CI overlaid.

        Pass ``stats={column: {"mean": .., "confidence_interval": (lo, hi)}}`` to
        skip ``compute_statistics`` (regenerate-from-saved). Returns list of (fig, axes).
        """
        data_frames = self._prepare_data_frames(data)
        variables_to_plot = self._select_vars(
            next(iter(data_frames.values())), variables_to_plot
        )

        results = []
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
                if stats is not None:
                    s = stats.get(column, {})
                else:
                    s = ds.compute_statistics(
                        column_name=column, window_size=window_size
                    ).get(column, {})
                mu = s.get("mean")
                ci = s.get("confidence_interval", (None, None))

                axes[j].plot(df["time"], df[column], label=column, alpha=0.7)
                if mu is not None:
                    axes[j].axhline(y=mu, color="red", linestyle="-", label=f"Mean={mu:.4g}")
                if ci[0] is not None and ci[1] is not None:
                    axes[j].fill_between(
                        df["time"], ci[0], ci[1], color="red", alpha=0.15, label="95 % CI"
                    )
                axes[j].set_xlabel("Time")
                axes[j].set_ylabel(column)
                axes[j].set_title(column)
                axes[j].legend(fontsize="small")
                axes[j].grid(True)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Trace with Mean — {self.format_dataset_name(dataset_name)}", fontsize=16
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            results.append(self._finalize(
                fig, axes,
                default_name=f"trace_with_mean_{self.format_dataset_name(dataset_name)}.png",
                save=save, show=show, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            ))
        return results

    # ------------------------------------------------------------------ #
    #  Ensemble trace plots                                                 #
    # ------------------------------------------------------------------ #

    def ensemble_trace_plot(self, data, variables_to_plot=None, *, save=False,
                            show=False, output_dir=None, overwrite=False, dpi=150):
        """Overlay traces from all members, one figure per variable. Returns list of (fig, axes)."""
        data_frames = self._prepare_data_frames(data)
        variables_to_plot = self._select_vars(
            next(iter(data_frames.values())), variables_to_plot
        )

        results = []
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
            results.append(self._finalize(
                fig, ax, default_name=f"ensemble_trace_{var}.png",
                save=save, show=show, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            ))
        return results

    def ensemble_trace_plot_with_mean(self, data, variables_to_plot=None,
                                      window_size=None, *, means=None, save=False,
                                      show=False, output_dir=None, overwrite=False, dpi=150):
        """Overlay member traces with per-member block mean.

        Pass ``means={dataset_name: {var: mean}}`` to skip ``compute_statistics``.
        Returns list of (fig, axes).
        """
        data_frames = self._prepare_data_frames(data)
        variables_to_plot = self._select_vars(
            next(iter(data_frames.values())), variables_to_plot
        )

        results = []
        for var in variables_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            for dataset_name, df in data_frames.items():
                if means is not None:
                    mu = means.get(dataset_name, {}).get(var)
                else:
                    mu = DataStream(df).compute_statistics(
                        column_name=var, window_size=window_size
                    ).get(var, {}).get("mean")
                ax.plot(df["time"], df[var], label=dataset_name, alpha=0.6)
                if mu is not None:
                    ax.axhline(y=mu, linestyle="--", linewidth=1.0,
                               label=f"{dataset_name} mean={mu:.4g}")
            ax.set_xlabel("Time")
            ax.set_ylabel(var)
            ax.set_title(f"Ensemble Traces with Means — {var}")
            ax.legend(fontsize="small")
            ax.grid(True)
            plt.tight_layout()
            results.append(self._finalize(
                fig, ax, default_name=f"ensemble_trace_with_mean_{var}.png",
                save=save, show=show, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            ))
        return results

    # ------------------------------------------------------------------ #
    #  Steady-state detection plots (single DataStream)                    #
    # ------------------------------------------------------------------ #

    def steady_state_automatic_plot(self, data, variables_to_plot=None, batch_size=10,
                                    start_time=0.0, method="std", threshold=None,
                                    robust=True, *, ss_starts=None, save=False, show=False,
                                    output_dir=None, overwrite=False, dpi=150):
        """Auto-detect steady-state start per variable and annotate.

        Pass ``ss_starts={column: ss_start_time}`` to skip trimming
        (regenerate-from-saved). Returns list of (fig, axes).
        """
        data_frames = self._prepare_data_frames(data)
        variables_to_plot = self._select_vars(
            next(iter(data_frames.values())), variables_to_plot
        )

        results = []
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

                if ss_starts is not None:
                    ss_start = ss_starts.get(column)
                else:
                    trimmed_ds = self._trim_datastream(
                        DataStream(df), column, method=method, batch_size=batch_size,
                        start_time=start_time, threshold=threshold, robust=robust,
                    )
                    ss_start = (
                        trimmed_ds.data["time"].iloc[0]
                        if trimmed_ds is not None and not trimmed_ds.data.empty
                        else None
                    )

                if ss_start is not None:
                    after_ss = signal[time >= ss_start]
                    mu, sigma = after_ss.mean(), after_ss.std()
                    t_end = time.max()
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(x=ss_start, color="r", linestyle="--", label="Steady-State Start")
                    ax.plot([ss_start, t_end], [mu, mu], color="g", linewidth=2, label="Mean (Post-SS)")
                    # ±1/2/3 std bands only for the std / QuantileTrimStrategy criterion,
                    # drawn from the steady-state start to the end of the trace.
                    if method == "std":
                        self._draw_std_bands(ax, time[time >= ss_start], mu, sigma)
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    logger.info("%s: no steady state detected — plotting full signal.", column)

                ax.set_title(column)
                ax.set_xlabel("Time")
                ax.set_ylabel(column)
                ax.legend(fontsize="small")
                ax.grid(True)

            for k in range(idx + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Steady-State Detection — {self.format_dataset_name(dataset_name)}", fontsize=14
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            results.append(self._finalize(
                fig, axes,
                default_name=f"steady_state_auto_{self.format_dataset_name(dataset_name)}.png",
                save=save, show=show, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            ))
        return results

    def steady_state_plot(self, data, variables_to_plot=None, steady_state_start=None,
                          *, show_std_bands=False, save=False, show=False,
                          output_dir=None, overwrite=False, dpi=150):
        """Annotate steady state from a user-supplied start (float or {var: float}).

        Use this after :meth:`~quends.DataStream.trim` has identified the
        steady-state start: pass ``steady_state_start=trimmed.trim_metadata
        ["sss_start"]`` so the plot uses the exact same point as the trim. The
        post-steady-state mean is drawn over that region; pass
        ``show_std_bands=True`` to also draw the ±1/2/3 std bands (appropriate
        for the std / QuantileTrimStrategy criterion). Returns list of (fig, axes).
        """
        data_frames = self._prepare_data_frames(data)
        variables_to_plot = self._select_vars(
            next(iter(data_frames.values())), variables_to_plot
        )

        results = []
        for dataset_name, df in data_frames.items():
            if "time" not in df.columns:
                raise ValueError(f"DataFrame for '{dataset_name}' is missing a 'time' column.")

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
                manual_ss = (
                    steady_state_start.get(column, None)
                    if isinstance(steady_state_start, dict)
                    else steady_state_start
                )

                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    mu, sigma = after_ss.mean(), after_ss.std()
                    t_end = time.max()
                    ax.plot(time, signal, label=column, alpha=0.7)
                    ax.axvline(x=manual_ss, color="r", linestyle="--", label="Steady-State Start")
                    ax.plot([manual_ss, t_end], [mu, mu], color="g", linewidth=2, label="Mean (Post-SS)")
                    if show_std_bands:
                        self._draw_std_bands(ax, time[time >= manual_ss], mu, sigma)
                else:
                    ax.plot(time, signal, label=column, alpha=0.7)
                    logger.info("%s: no steady state start provided — plotting raw signal.", column)

                ax.set_title(column)
                ax.set_xlabel("Time")
                ax.set_ylabel(column)
                ax.legend(fontsize="small")
                ax.grid(True, alpha=0.3)

            for k in range(j + 1, len(axes)):
                fig.delaxes(axes[k])

            plt.suptitle(
                f"Steady-State (Manual) — {self.format_dataset_name(dataset_name)}", fontsize=14
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            results.append(self._finalize(
                fig, axes,
                default_name=f"steady_state_manual_{self.format_dataset_name(dataset_name)}.png",
                save=save, show=show, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            ))
        return results

    # ------------------------------------------------------------------ #
    #  ACF plots                                                            #
    # ------------------------------------------------------------------ #

    def plot_acf(self, data, alpha=0.05, column=None, ax=None, *, acf_values=None,
                 save=False, show=False, filename=None, output_dir=None,
                 overwrite=False, dpi=150):
        """Plot the autocorrelation function.

        Pass ``acf_values`` (precomputed array) to skip ``acf()``. If ``ax`` is
        given the stem is drawn into it (no save/show); otherwise a new figure is
        created and returned as (fig, ax).
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

        if acf_values is None:
            nlags = max(1, int(n / 3))
            acf_values = acf(filtered, nlags=nlags, fft=False)
        conf = norm.ppf(1 - alpha / 2) / np.sqrt(n)

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(12, 6))

        ax.stem(range(len(acf_values)), acf_values, basefmt=" ")
        ax.axhline(conf, color="red", linestyle="--", label=f"95 % CI upper: {conf:.3f}")
        ax.axhline(-conf, color="red", linestyle="--", label=f"95 % CI lower: {-conf:.3f}")
        ax.set_title(f"ACF — '{column}'")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

        if own_fig:
            plt.tight_layout()
            return self._finalize(
                fig, ax, default_name=f"acf_{column}.png", save=save, show=show,
                filename=filename, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            )
        return None, ax

    def plot_acf_ensemble(self, ensemble_obj, alpha=0.05, column=None, *, save=False,
                          show=False, filename=None, output_dir=None, overwrite=False, dpi=150):
        """ACF grid, one subplot per member. Returns (fig, axes)."""
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
        return self._finalize(
            fig, axes, default_name="ensemble_acf.png", save=save, show=show,
            filename=filename, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
        )

    # ------------------------------------------------------------------ #
    #  Ensemble steady-state plots                                          #
    # ------------------------------------------------------------------ #

    def ensemble_steady_state_automatic_plot(self, ensemble_obj, variables_to_plot=None,
                                             batch_size=10, start_time=0.0, method="std",
                                             threshold=None, robust=True, *, ss_starts=None,
                                             save=False, show=False, filename=None,
                                             output_dir=None, overwrite=False, dpi=150):
        """Auto-detect steady state per member, one subplot each. Returns (fig, axes).

        Pass ``ss_starts={member_index: {var: ss_start}}`` to skip trimming.
        """
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(math.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.data
            vars_plot = self._select_vars(df, variables_to_plot)
            ax = axes[i]
            time = df["time"]

            for var in vars_plot:
                signal = df[var]
                if ss_starts is not None:
                    ss_start = ss_starts.get(i, {}).get(var)
                else:
                    trimmed = self._trim_datastream(
                        ds, var, method=method, batch_size=batch_size,
                        start_time=start_time, threshold=threshold, robust=robust,
                    )
                    ss_start = (
                        trimmed.data["time"].iloc[0]
                        if trimmed is not None and not trimmed.data.empty
                        else None
                    )
                if ss_start is not None:
                    after_ss = signal[time >= ss_start]
                    mu, sigma = after_ss.mean(), after_ss.std()
                    ax.plot(time, signal, label=var, alpha=0.7)
                    ax.axvline(x=ss_start, color="r", linestyle="--", label="SS Start")
                    ax.axhline(y=mu, color="g", linestyle="-", label="Mean")
                    self._draw_std_bands(ax, time[time >= ss_start], mu, sigma)
                else:
                    ax.plot(time, signal, label=var, alpha=0.7)
                    logger.info("Member %d / %s: no SS detected.", i, var)

            ax.set_title(f"Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(", ".join(vars_plot))
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._finalize(
            fig, axes, default_name="ensemble_steady_state_auto.png", save=save, show=show,
            filename=filename, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
        )

    def ensemble_steady_state_plot(self, ensemble_obj, variables_to_plot=None,
                                   steady_state_start=None, *, save=False, show=False,
                                   filename=None, output_dir=None, overwrite=False, dpi=150):
        """Annotate each member with a user-supplied SS start. Returns (fig, axes)."""
        n_members = len(ensemble_obj.data_streams)
        ncols = min(3, n_members)
        nrows = int(np.ceil(n_members / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for i, ds in enumerate(ensemble_obj.data_streams):
            df = ds.data
            vars_plot = self._select_vars(df, variables_to_plot)
            ax = axes[i]
            time = df["time"]

            for var in vars_plot:
                signal = df[var]
                manual_ss = (
                    steady_state_start.get(var, None)
                    if isinstance(steady_state_start, dict)
                    else steady_state_start
                )
                if manual_ss is not None:
                    after_ss = signal[time >= manual_ss]
                    mu, sigma = after_ss.mean(), after_ss.std()
                    ax.plot(time, signal, label=var, alpha=0.7)
                    ax.axvline(x=manual_ss, color="r", linestyle="--", label="SS Start")
                    ax.axhline(y=mu, color="g", linestyle="-", label="Mean")
                    self._draw_std_bands(ax, time[time >= manual_ss], mu, sigma)
                else:
                    ax.plot(time, signal, label=var, alpha=0.7)
                    logger.info("Member %d / %s: no SS start provided.", i, var)

            ax.set_title(f"Member {i}")
            ax.set_xlabel("Time")
            ax.set_ylabel(", ".join(vars_plot))
            ax.legend(fontsize="small")
            ax.grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return self._finalize(
            fig, axes, default_name="ensemble_steady_state_manual.png", save=save, show=show,
            filename=filename, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
        )

    # ------------------------------------------------------------------ #
    #  Ensemble member + average plots                                      #
    # ------------------------------------------------------------------ #

    def plot_ensemble(self, ensemble_obj, variables_to_plot=None, *, avg_df=None,
                      save=False, show=False, filename=None, output_dir=None,
                      overwrite=False, dpi=150):
        """Members + ensemble average, 2-column grid. Returns (fig, axes).

        Pass ``avg_df`` (a precomputed average DataFrame) to skip
        ``compute_average_ensemble``.
        """
        return self.plot_ensemble_with_average(
            ensemble_obj, variables_to_plot=variables_to_plot, avg_df=avg_df,
            save=save, show=show, filename=filename or "ensemble_members_and_average.png",
            output_dir=output_dir, overwrite=overwrite, dpi=dpi,
            condensed_legend=False, y_range=None,
        )

    def plot_ensemble_with_average(self, ensemble_obj, variables_to_plot=None, *,
                                   avg_df=None, condensed_legend=False, y_range=None,
                                   save=False, show=False, filename=None, output_dir=None,
                                   overwrite=False, dpi=150):
        """Members + ensemble average with optional condensed legend / y-range.

        Pass ``avg_df`` to skip ``compute_average_ensemble``. Returns (fig, axes).
        """
        member_dfs = {
            f"Member {i}": ds.data for i, ds in enumerate(ensemble_obj.data_streams)
        }
        if avg_df is None:
            avg_df = ensemble_obj.compute_average_ensemble().data
        member_dfs["Ensemble Average"] = avg_df

        all_cols = avg_df.columns.tolist()
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
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 7 * nrows), squeeze=False)
        axes = axes.flatten()

        for idx, var in enumerate(vars_to_plot):
            ax = axes[idx]
            first_drawn = False
            for name, df in member_dfs.items():
                if name == "Ensemble Average":
                    ax.plot(df["time"], df[var], label="Ensemble Average",
                            color="black", linewidth=2.5, zorder=5)
                else:
                    if condensed_legend:
                        label = "Individual Members" if not first_drawn else None
                    else:
                        label = name
                    ax.plot(df["time"], df[var], label=label, alpha=0.3, linewidth=1.0)
                    first_drawn = True
            ax.set_title(var, fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(var, fontsize=12)
            ax.grid(True, alpha=0.3)
            if y_range is not None:
                ax.set_ylim(y_range)

        for j in range(n_vars, len(axes)):
            fig.delaxes(axes[j])

        handles, labels = axes[0].get_legend_handles_labels()
        if condensed_legend:
            seen, unique = set(), []
            for h, l in zip(handles, labels):
                if l is not None and l not in seen:
                    seen.add(l)
                    unique.append((h, l))
            handles, labels = zip(*unique) if unique else ([], [])

        legend_ncol = min(len(labels), 4) if labels else 1
        legend_y = -0.05 if n_vars == 1 else -0.08
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
                   ncol=legend_ncol, fontsize="small", frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        return self._finalize(
            fig, axes, default_name="ensemble_with_average.png", save=save, show=show,
            filename=filename, output_dir=output_dir, overwrite=overwrite, dpi=dpi,
        )
