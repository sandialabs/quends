import math
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller

from .history import DataStreamHistory, DataStreamHistoryEntry
from .utils import (
    SCHEMA_VERSION,
    StatsResult,
    _compute_ess,
    _estimate_tau_int_from_series,
    _geyer_ess_on_blocks,
    _resolve_columns,
    autotune_blocks,
    confidence_multiplier,
    power_law_model,
    to_native_types,
)

"""
Usage signature for the new trimming workflow::

    strategy = SomeTrimStrategy(kwargs...)
    trim_operation = TrimDataStreamOperation(strategy=strategy)
    data_stream = DataStream(data)
    trimmed_data_stream = trim_operation(data_stream, column_name)
"""


TAU_INT_LAG_CUTOFF_WARNING_RATIO = 0.5


def _tau_int_metadata_warning(tau_int: float, n_samples: int) -> Optional[str]:
    """Mirror the tau_int lag-cutoff warning as result metadata."""
    if not np.isfinite(tau_int) or n_samples < 3:
        return None
    nlags = max(1, min(n_samples // 4, 2000))
    if tau_int < TAU_INT_LAG_CUTOFF_WARNING_RATIO * nlags:
        return None
    return (
        "The computed signal decorrelation time is large compared to the "
        "max lag in the computation of the autocorrelation. Results may "
        f"be inaccurate. Estimated tau_int={tau_int:.2f}, nlags={nlags}."
    )


def _coerce_history(history: Optional[Any] = None) -> DataStreamHistory:
    """Return an independent typed history object from typed or legacy input."""

    if history is None:
        return DataStreamHistory()
    if isinstance(history, DataStreamHistory):
        return history.copy()

    entries = []
    for entry in history:
        if isinstance(entry, DataStreamHistoryEntry):
            entries.append(entry)
        elif isinstance(entry, dict):
            entries.append(
                DataStreamHistoryEntry(
                    operation_name=entry.get("operation", "unknown"),
                    parameters=entry.get("options", {}),
                )
            )
        else:
            entries.append(
                DataStreamHistoryEntry(
                    operation_name=type(entry).__name__,
                    parameters={"value": entry},
                )
            )
    return DataStreamHistory(entries)


class DataStream:

    def __init__(self, data: Any, history: Optional[DataStreamHistory] = None) -> None:
        """Wrap a pandas DataFrame of time-series data.

        ``data`` must be a :class:`pandas.DataFrame` (or something convertible to
        one — a dict/array/Series is coerced); anything else raises ``TypeError``
        early with a clear message rather than failing cryptically later.

        A ``time`` column is **not** required at construction — column-wise
        statistics work without it — but steady-state trimming and ensemble
        averaging do require one and will raise a clear error if it is missing
        (see :meth:`trim` / :mod:`quends.base.trim`).
        """
        if isinstance(data, pd.DataFrame):
            df = data
        elif data is None:
            raise TypeError(
                "DataStream(data): data must be a pandas DataFrame, got None."
            )
        else:
            try:
                df = pd.DataFrame(data)
            except Exception as exc:  # noqa: BLE001 - re-raised as a clear TypeError
                raise TypeError(
                    "DataStream(data): data must be a pandas DataFrame (or something "
                    f"convertible to one); got {type(data).__name__} ({exc})."
                )
        self._data = df
        self._history = _coerce_history(history)

    @property
    def data(self) -> Any:
        """The underlying pandas DataFrame."""
        return self._data

    @property
    def df(self) -> Any:
        """Backward-compatible alias for the underlying pandas DataFrame."""
        return self._data

    @df.setter
    def df(self, value: Any) -> None:
        self._data = value

    @property
    def history(self) -> DataStreamHistory:
        return self._history

    def head(self, n: int = 5) -> Any:
        return self.data.head(n)

    def __len__(self) -> int:
        return len(self.data)

    def variables(self):
        """
        Return all column names in the underlying DataFrame (including 'time').

        To obtain only signal columns use::

            [c for c in ds.variables() if c != "time"]

        Returns
        -------
        pandas.Index
            All column names in ``self.data``.
        """
        return self.data.columns

    def trim(
        self,
        column_name=None,
        *,
        method: str = "std",
        window_size: int = 10,
        start_time: float = 0.0,
        threshold: Optional[float] = None,
        robust: bool = True,
        **strategy_kwargs: Any,
    ) -> "DataStream":
        """Trim this stream to its steady state and return a new ``DataStream``.

        Convenience one-liner over :func:`quends.base.trim.build_trim_strategy` +
        :class:`~quends.base.trim.TrimDataStreamOperation` (the explicit/low-level
        path still works exactly as before — this just wraps it).

        Parameters
        ----------
        column_name : str, optional
            Column to detect steady state on. If ``None`` and the stream has a
            single non-``time`` column, that column is used automatically.
        method : str
            ``"std"`` | ``"threshold"`` | ``"rolling_variance"`` |
            ``"self_consistent"`` | ``"iqr"`` | ``"mean_variation"``.
        window_size, start_time, threshold, robust :
            Strategy parameters (see ``build_trim_strategy``).
        **strategy_kwargs :
            Extra attributes set on the strategy (e.g. ``drop_leading_nonpositive=False``).

        Returns
        -------
        DataStream
            The trimmed stream (empty if no steady state was detected).
        """
        # Local import avoids a circular import at module load time.
        from .trim import TrimDataStreamOperation, build_trim_strategy

        if column_name is None:
            cols = [c for c in self.data.columns if c != "time"]
            if len(cols) != 1:
                raise ValueError(
                    "column_name must be specified when the stream does not have "
                    f"exactly one non-'time' column (found {cols})."
                )
            column_name = cols[0]

        strategy = build_trim_strategy(
            method=method,
            window_size=window_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )
        for key, value in strategy_kwargs.items():
            setattr(strategy, key, value)
        return TrimDataStreamOperation(strategy=strategy)(self, column_name=column_name)

    def compute_statistics(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        confidence_level: float = 0.95,
        ci_method: str = "normal",
    ):
        """
        Aggregate statistics for each column using autotuned independent block means.

        Window selection and block-mean computation go through :meth:`_process_column`
        → :func:`~quends.base.utils.autotune_blocks` (the single canonical helper
        shared with the ensemble pipeline).

        Parameters
        ----------
        column_name : str or list or None
        ddof : int
        method : {'non-overlapping', 'sliding'}
            Block type.  Independence autotuning always uses non-overlapping blocks
            regardless of this setting.
        window_size : int or None
            User-supplied window; triggers autotune when ``None``.
        confidence_level : float
            Two-sided confidence level for the CI.  Default ``0.95``.
        ci_method : {'normal', 't'}
            CI quantile family.  Default ``'normal'`` (preserves the historical
            ``1.96`` multiplier exactly for backward compatibility).  When ``'t'``,
            uses Student's *t* with ``dof = max(1, se_effective_n - 1)``.

        Returns
        -------
        dict
            ``{col: {…statistics…}}`` with the following canonical keys per column:

            ``mean``, ``mean_uncertainty`` (SEM), ``variance``, ``confidence_interval``,
            ``standard_deviation`` (standard deviation of the independent samples in the data stream), ``pm_std``,
            ``effective_sample_size`` (Geyer ESS on raw series), ``window_size``,
            ``n_short_averages`` (number of block means), ``ess_blocks``
            (Geyer ESS on block means), ``se_effective_n``, ``se_method``,
            ``independence_status``, ``independent``,
            ``ljungbox_lags`` (list), ``ljungbox_pvalues`` (list),
            ``ljungbox_pvalue`` (scalar min — convenience alias matching ensemble output),
            ``ci_method``, ``confidence_level``,
            ``warning`` (if applicable).

            On error: ``{col: {"error": "…"}}``
        """
        stats = {}
        cols = self._get_columns(column_name)
        ess_res = self.effective_sample_size(column_names=column_name)
        result_warnings = []

        for col in cols:
            series = self.data[col].dropna()
            if series.empty:
                stats[col] = {"error": f"No data available for column '{col}'"}
                continue

            # Window selection and block-mean computation via _process_column,
            # which is the single autotune entry point for both DataStream and
            # the ensemble pipeline.
            _, ab = self._process_column(
                series,
                estimated_window=window_size,
                method=method,
                min_blocks=2,  # continue until < 2 blocks (original behaviour)
                max_iter=20,
            )
            w = ab["window_size"]
            status = ab["independence_status"]
            block_means = ab["blocks"]
            lb = {"lags": ab["ljungbox_lags"], "pvalues": ab["ljungbox_pvalues"]}
            n_blocks = ab["n_blocks"]
            tau_int_warning = _tau_int_metadata_warning(
                ab.get("tau_int", float("nan")),
                len(series),
            )
            column_warnings = [tau_int_warning] if tau_int_warning else []

            if n_blocks < 1:
                stats[col] = {
                    "error": f"No block means produced (window_size={w}).",
                    "window_size": int(w),
                }
                continue

            mu = float(np.mean(block_means))
            var_val = float(np.var(block_means, ddof=ddof)) if n_blocks >= 2 else np.nan
            sd = float(np.std(block_means, ddof=ddof)) if n_blocks >= 2 else np.nan
            ess_blocks = _geyer_ess_on_blocks(block_means)

            # SE rule based on independence status
            warning = None
            if status == "independent":
                eff_n_for_se = float(n_blocks)
                se_method = "iid_blocks"
            elif status == "best_p":
                eff_n_for_se = float(max(1.0, n_blocks))
                se_method = "iid_blocks_best_p"
                warning = "Block means did not pass Ljung-Box; using best-p window."
            elif status == "user_window":
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks"
                warning = "User window; SE via Geyer ESS on block means."
            else:
                eff_n_for_se = float(max(1.0, ess_blocks))
                se_method = "ess_blocks_fallback"
                warning = "Too few blocks for independence test; SE via Geyer ESS."

            se = (
                float(sd / np.sqrt(eff_n_for_se))
                if np.isfinite(sd) and n_blocks >= 2
                else np.nan
            )

            # CI multiplier (defaults preserve historical 1.96 exactly).
            ci_dof = max(1, int(round(eff_n_for_se)) - 1) if ci_method == "t" else None
            ci_mult = confidence_multiplier(
                confidence_level=confidence_level, method=ci_method, dof=ci_dof
            )
            ci = (
                (float(mu - ci_mult * se), float(mu + ci_mult * se))
                if np.isfinite(se)
                else (np.nan, np.nan)
            )

            ess_entry = ess_res.get("results", {}).get(col, {})
            ess_val = (
                ess_entry.get("effective_sample_size")
                if isinstance(ess_entry, dict)
                else ess_entry
            )

            pvals = lb.get("pvalues", [])
            entry = {
                "mean": mu,
                "mean_uncertainty": se,
                "variance": var_val,
                "confidence_interval": ci,
                "standard_deviation": sd,
                "pm_std": (mu - se, mu + se) if np.isfinite(se) else (np.nan, np.nan),
                "effective_sample_size": int(ess_val) if ess_val is not None else None,
                "window_size": int(w),
                "n_short_averages": int(n_blocks),
                "ess_blocks": float(ess_blocks),
                "se_effective_n": float(eff_n_for_se),
                "se_method": se_method,
                "independence_status": status,
                "independent": bool(ab["independent"]),
                "ljungbox_lags": lb.get("lags", []),
                "ljungbox_pvalues": pvals,
                # Scalar convenience key (min of all tested lags) — mirrors
                # the ljungbox_pvalue key in the ensemble Technique-1 output.
                "ljungbox_pvalue": float(min(pvals)) if pvals else float("nan"),
                # CI provenance
                "ci_method": ci_method,
                "confidence_level": float(confidence_level),
            }
            if warning:
                entry["warning"] = warning
                column_warnings.append(warning)
            if column_warnings:
                entry["metadata"] = {"warnings": column_warnings}
                result_warnings.extend(
                    {"column": col, "message": message} for message in column_warnings
                )
            stats[col] = entry

        # Return a StatsResult: behaves exactly like the historical
        # {column: {...}} dict (so res[col]["mean_uncertainty"] and equality
        # checks are unchanged) but carries run-level provenance in .metadata.
        metadata = {
            "estimator": "single",
            "columns": list(stats.keys()),
            "total_samples": int(len(self.data)),
            "schema_version": SCHEMA_VERSION,
            "warnings": result_warnings,
        }
        return StatsResult(to_native_types(stats), metadata=to_native_types(metadata))

    def cumulative_statistics(
        self, column_name=None, method="non-overlapping", window_size=None
    ):
        """
        Generate cumulative mean and uncertainty time series for each column.

        Returns per-column cumulative arrays plus ``window_size``.

        Notes
        -----
        ``cumulative_uncertainty`` is the expanding **standard deviation** of the
        processed series, while ``standard_error`` is the expanding SEM
        (std / sqrt(count)). Use ``standard_error`` for uncertainty-on-the-mean.
        """
        results = {}
        for col in self._get_columns(column_name):
            column_data = self.data[col].dropna()
            if column_data.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue
            time_values = self._time_values_for_series(column_data)
            proc_data, ab = self._process_column(
                column_data, window_size, method, time_values=time_values
            )
            est_win = ab["window_size"]
            cumulative_mean = proc_data.expanding().mean()
            cumulative_std = proc_data.expanding().std()
            count = proc_data.expanding().count()
            standard_error = cumulative_std / np.sqrt(count)
            results[col] = {
                "cumulative_mean": cumulative_mean.tolist(),
                "cumulative_uncertainty": cumulative_std.tolist(),
                "standard_error": standard_error.tolist(),
                "window_size": int(est_win),
            }

        return to_native_types(results)

    def additional_data(
        self,
        column_name=None,
        ddof=1,
        method="sliding",
        window_size=None,
        reduction_factor=0.1,
    ):
        """
        Estimate additional sample size needed to reduce SEM by `reduction_factor` via power-law fit.

        Returns model parameters and sample projections.

        Notes
        -----
        The power law is currently fit to ``cumulative_statistics``'
        ``cumulative_uncertainty`` series. See ``cumulative_statistics`` — that
        key holds the expanding standard deviation, not the SEM; fitting a
        shrinking-SEM power law to it is a known limitation (see AUDIT_REPORT H2).
        """
        stats = self.cumulative_statistics(
            column_name, method=method, window_size=window_size
        )
        results = {}
        columns = self._get_columns(column_name)
        for col in columns:
            if "standard_error" not in stats.get(col, {}):
                results[col] = {"error": f"No cumulative SEM data for column '{col}'"}
                continue
            est_win = stats[col].get(
                "window_size", 1
            )  # already computed by cumulative_statistics
            # Fit the power law to the standard error of the mean (SEM), which
            # genuinely shrinks ~ A/n^p — NOT the expanding standard deviation
            # (which converges to a constant). See AUDIT_REPORT H2.
            cum_sem = np.array(stats[col]["standard_error"])
            n_current = len(cum_sem)
            cumulative_count = np.arange(1, n_current + 1)
            mask = np.isfinite(cum_sem)
            valid_count = cumulative_count[mask]
            valid_sem = cum_sem[mask]
            if len(valid_count) < 2:
                results[col] = {"error": "Not enough valid data points for fitting."}
                continue

            popt, _ = curve_fit(power_law_model, valid_count, valid_sem, p0=[1.0, 0.5])
            A_est, p_est = popt
            p_est = abs(p_est)
            current_sem = power_law_model(n_current, A_est, p_est)
            target_sem = (1 - reduction_factor) * current_sem
            n_target = (A_est / target_sem) ** (1 / p_est)
            additional_samples = n_target - n_current
            if method == "non-overlapping":
                additional_samples *= est_win
            results[col] = {
                "A_est": float(A_est),
                "p_est": float(p_est),
                "n_current": int(n_current),
                "current_sem": float(current_sem),
                "target_sem": float(target_sem),
                "n_target": float(n_target),
                "additional_samples": int(math.ceil(additional_samples)),
                "window_size": int(est_win),
            }

        return to_native_types(results)

    # ------ Helper functions --------
    def is_stationary(self, columns):
        """
        Perform Augmented Dickey-Fuller test for each specified column.

        Parameters
        ----------
        columns : str or list of str

        Returns
        -------
        dict
            {column: True if stationary (p < 0.05), else False}
        """
        if isinstance(columns, str):
            columns = [columns]
        results = {}
        for column in columns:
            try:
                p_value = adfuller(self.data[column].dropna(), autolag="AIC")[1]
                results[column] = bool(p_value < 0.05)
            except Exception:
                results[column] = False
        return results

    def _get_columns(self, column_name):
        """
        Resolve `column_name` parameter into a list of valid DataFrame columns.

        Returns
        -------
        list of str
        """
        if column_name is None:
            return [col for col in self.data.columns if col != "time"]
        return [column_name] if isinstance(column_name, str) else column_name

    def _estimate_window(self, col, column_data, window_size):
        """
        Legacy wrapper — return user-supplied ``window_size`` or autotune via
        ``_autotune_window_size``.

        .. deprecated::
            No live callers remain inside the codebase.  The main pipeline now
            goes through :meth:`_process_column`, which calls
            :func:`~quends.base.utils.autotune_blocks` directly.  This method
            is retained only for any external code that may call it.
        """
        if window_size is not None:
            return int(window_size)
        w, _ = self._autotune_window_size(
            col=col,
            series=column_data,
            method="non-overlapping",
            alpha=0.05,
            lag_set=(5, 10),
            B_min=15,
            c0=2.0,
            max_iter=20,
            w_min=5,
        )
        return int(w)

    def _process_column(
        self,
        column_data,
        estimated_window: Optional[int],
        method: str,
        time_values: Optional[np.ndarray] = None,
        # --- autotune parameters forwarded to autotune_blocks ---
        alpha: float = 0.05,
        lag_set=(5, 10),
        B_min: int = 15,
        min_blocks: int = 2,
        max_iter: int = 25,
        w_min: int = 5,
        c0: float = 2.0,
    ) -> Tuple[pd.Series, dict]:
        """
        Autotune block window and transform a 1D series into block means.

        Window determination always uses non-overlapping blocks + Ljung-Box
        (LB on sliding means is unreliable for independence testing).  When
        *estimated_window* is ``None`` the window is chosen by
        :func:`~quends.base.utils.autotune_blocks`; when an integer is
        supplied the autotune is skipped but the LB diagnostic is still run.

        Parameters
        ----------
        column_data : pandas.Series
            Raw series values (dropna'd before calling).
        estimated_window : int or None
            Starting window hint.  ``None`` triggers full autotune.
        method : {'non-overlapping', 'sliding'}
            Output format.  Independence autotune always uses
            ``non-overlapping`` regardless of this setting.
        time_values : np.ndarray or None
            If provided, used as block-centre indices in the returned Series.
        alpha, lag_set, B_min, min_blocks, max_iter, w_min, c0 :
            Forwarded verbatim to :func:`~quends.base.utils.autotune_blocks`.

        Returns
        -------
        (block_means_series : pd.Series, autotune_result : dict)
            *autotune_result* is the full dict returned by
            :func:`~quends.base.utils.autotune_blocks`; it always describes
            non-overlapping blocks regardless of *method*.
        """
        if method not in ("non-overlapping", "sliding"):
            raise ValueError("Invalid method. Choose 'sliding' or 'non-overlapping'.")

        if time_values is not None:
            time_values = np.asarray(time_values, dtype=float)
            if time_values.size != len(column_data):
                time_values = None

        x = np.asarray(column_data.values, dtype=float)

        # Independence autotune always on non-overlapping blocks.
        ab = autotune_blocks(
            x,
            window_size=estimated_window,
            method="non-overlapping",
            alpha=alpha,
            lag_set=lag_set,
            B_min=B_min,
            min_blocks=min_blocks,
            max_iter=max_iter,
            w_min=w_min,
            c0=c0,
        )
        w = ab["window_size"]

        if method == "sliding":
            if time_values is not None:
                series = pd.Series(column_data.values, index=time_values)
            else:
                series = column_data
            return series.rolling(window=int(w)).mean().dropna(), ab

        # non-overlapping — reuse the blocks already computed by autotune_blocks
        blocks = ab["blocks"]
        n_blocks = int(blocks.size)
        if n_blocks < 1:
            return pd.Series([], dtype=float), ab

        if time_values is not None:
            n_usable = n_blocks * w
            t = time_values[:n_usable].reshape(n_blocks, w)
            idx = t.mean(axis=1)
        else:
            idx = (np.arange(n_blocks) * w) + (w // 2)

        return pd.Series(blocks, index=idx), ab

    def _time_values_for_series(self, series: pd.Series) -> Optional[np.ndarray]:
        """Return the 'time' column values aligned to `series.index`, or None."""
        if "time" not in self.data.columns:
            return None
        try:
            return self.data.loc[series.index, "time"].to_numpy(dtype=float)
        except Exception:
            return None

    # ----------- Window autotune (tau_int + Ljung-Box) ----------------

    def _estimate_tau_int(self, series: pd.Series) -> float:
        """
        Estimate tau_int from raw series ACF using Geyer positive-pair truncation.

        Delegates to ``_estimate_tau_int_from_series`` in ``utils``.
        """
        return _estimate_tau_int_from_series(
            np.asarray(series.dropna().values, dtype=float)
        )

    def _autotune_window_size(
        self,
        col: str,
        series: pd.Series,
        method: str = "non-overlapping",
        alpha: float = 0.05,
        lag_set=(5, 10),
        B_min: int = 15,
        c0: float = 2.0,
        max_iter: int = 20,
        w_min: int = 5,
    ):
        """
        Choose block window size via tau_int + Ljung-Box autotune.

        Thin wrapper around :func:`~quends.base.utils.autotune_blocks` for
        backward compatibility.  The canonical pipeline entry point is
        :meth:`_process_column`, which calls ``autotune_blocks`` directly.

        Returns
        -------
        (chosen_w : int, info : dict)
            ``info`` keys: ``status``, ``tau_int``, ``chosen_w``, ``passed``,
            ``lb`` (sub-dict with ``lags`` and ``pvalues`` lists).
        """
        x = np.asarray(series.dropna().values, dtype=float)
        result = autotune_blocks(
            x,
            window_size=None,
            method=method,
            alpha=alpha,
            lag_set=lag_set,
            B_min=B_min,
            min_blocks=2,  # match old DataStream: continue until < 2 blocks
            max_iter=max_iter,
            w_min=w_min,
            c0=c0,
        )
        info = {
            "status": result["independence_status"],
            "tau_int": result.get("tau_int", float("nan")),
            "chosen_w": result["window_size"],
            "passed": result["independent"],
            "lb": {
                "lags": result["ljungbox_lags"],
                "pvalues": result["ljungbox_pvalues"],
            },
        }
        return int(result["window_size"]), info

    # ----------- Block-level ESS + variance ----------------

    def get_block_effective_n(
        self,
        column_name: str,
        method: str = "non-overlapping",
        window_size: Optional[int] = None,
    ) -> dict:
        """
        Return Geyer ESS on block means for one column.

        Thin wrapper over :meth:`compute_statistics` — extracts the
        ``ess_blocks``, ``window_size``, and ``n_short_averages`` fields so
        that callers that only need block-level ESS info don't have to unpack
        the full statistics dict.

        Returns
        -------
        dict
            ``{"effective_n": float, "window_size": int, "n_blocks": int}``
        """
        stats = self.compute_statistics(
            column_name=column_name, method=method, window_size=window_size
        )
        v = stats.get(column_name, {})
        if "error" in v:
            return {
                "effective_n": float("nan"),
                "window_size": window_size or 0,
                "n_blocks": 0,
            }
        return {
            "effective_n": float(v["ess_blocks"]),
            "window_size": int(v["window_size"]),
            "n_blocks": int(v["n_short_averages"]),
        }

    def _variance(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ) -> dict:
        """
        Variance of block means.

        Thin wrapper over :meth:`compute_statistics` — extracts ``variance``,
        ``window_size``, and ``ess_blocks`` (returned as ``effective_n_blocks``
        for backward compatibility).
        """
        stats = self.compute_statistics(
            column_name=column_name, ddof=ddof, method=method, window_size=window_size
        )
        results = {}
        for col, v in stats.items():
            if "error" in v:
                results[col] = {
                    "variance": float("nan"),
                    "window_size": float("nan"),
                    "effective_n_blocks": float("nan"),
                }
            else:
                results[col] = {
                    "variance": v["variance"],
                    "window_size": v["window_size"],
                    "effective_n_blocks": v["ess_blocks"],
                }
        return results

    # ----------- ESS (classic and robust) ----------------
    def effective_sample_size(self, column_names=None, alpha=0.05):
        """
        Compute ESS using Geyer positive-pair truncation of the ACF.

        The integrated autocorrelation time ``tau_int`` is estimated by summing
        consecutive positive pairs of the normalised ACF and truncating as soon
        as a pair turns non-positive.  ESS is then ``n / tau_int``.

        Parameters
        ----------
        column_names : str or list of str or None
            Columns to compute ESS for; defaults to all non-time columns.
        alpha : float
            Reserved for API compatibility; not used in the Geyer estimator.

        Returns
        -------
        dict
            ``{'results': {col: ESS_int or message_dict}}``
        """

        columns = _resolve_columns(self.data, column_names)
        results = {col: _compute_ess(self.data, col, alpha) for col in columns}

        return {
            "results": to_native_types(results),
        }

    def estimate_tau_int(self, column_name=None):
        """
        Estimate the integrated autocorrelation time (tau_int) for specified columns.

        Parameters

        column_name : str or list or None
            Column(s) to compute tau_int for; defaults to all non-time columns.

        Returns
        -------
        dict
            {'results': {col: tau_int}}
        """
        columns = _resolve_columns(self.data, column_name)
        results = {}
        for col in columns:
            series = self.data[col].dropna()
            if series.empty:
                results[col] = {"error": f"No data available for column '{col}'"}
                continue

            tau_int = _estimate_tau_int_from_series(
                np.asarray(series.values, dtype=float)
            )
            results[col] = tau_int

        return {"results": to_native_types(results)}

    @staticmethod
    def robust_effective_sample_size(
        x,
        rank_normalize=True,
        min_samples=8,
        return_relative=False,
    ):
        """
        Compute a robust ESS via pairwise autocorrelations and optional rank-normalization.

        Parameters
        ----------
        x : array-like
        rank_normalize : bool
        min_samples : int
        return_relative : bool

        Returns
        -------
        float or tuple
            ESS (and ESS/n ratio if return_relative).
        """
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        n = len(x)
        if n < min_samples:
            return np.nan if not return_relative else (np.nan, np.nan)
        if np.all(x == x[0]):
            return float(n) if not return_relative else (float(n), 1.0)
        if rank_normalize:
            x = rankdata(x, method="average")
            x = (x - np.mean(x)) / np.std(x, ddof=0)
        else:
            x = x - np.mean(x)
        var = np.var(x, ddof=0)
        if var == 0:
            return float(n) if not return_relative else (float(n), 1.0)
        acorr = np.empty(n)
        acorr[0] = 1.0
        for lag in range(1, n):
            v1 = x[:-lag]
            v2 = x[lag:]
            acorr[lag] = np.dot(v1, v2) / ((n - lag) * var)
        s = 0.0
        t = 1
        while t + 1 < n:
            pair_sum = acorr[t] + acorr[t + 1]
            if pair_sum < 0:
                break
            s += pair_sum
            t += 2
        ess = n / (1 + 2 * s)
        ess = max(1.0, min(ess, n))
        if return_relative:
            return ess, ess / n
        return ess

    def ess_robust(
        self,
        column_names=None,
        rank_normalize=False,
        min_samples=8,
        return_relative=False,
    ):
        """
        Wrapper for `robust_effective_sample_size` over multiple columns.

        Parameters
        ----------
        column_names : str or list or None
        rank_normalize : bool
        min_samples : int
        return_relative : bool

        Returns
        -------
        dict
            {'results': {col: ESS or tuple}}
        """

        if column_names is None:
            column_names = [col for col in self.data.columns if col != "time"]
        elif isinstance(column_names, str):
            column_names = [column_names]
        results = {}
        for col in column_names:
            if col not in self.data.columns:
                results[col] = {"error": f"Column '{col}' not found."}
                continue
            x = self.data[col].dropna().values
            ess = self.robust_effective_sample_size(
                x,
                rank_normalize=rank_normalize,
                min_samples=min_samples,
                return_relative=return_relative,
            )
            results[col] = ess

        return {"results": to_native_types(results)}

    @staticmethod
    def normalize_data(df):
        """
        Min-Max normalize all signal columns (excluding 'time') to [0,1].

        Operates on a copy; the input DataFrame is not mutated.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        df = df.copy()
        scaler = MinMaxScaler()
        signal_columns = df.columns[1:]
        df[signal_columns] = df[signal_columns].astype(float)
        df[signal_columns] = scaler.fit_transform(df[signal_columns])
        return df
