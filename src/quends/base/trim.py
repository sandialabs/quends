from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as sts
import statsmodels.tsa.stattools as ststls
from matplotlib import pyplot as plt
from statsmodels.robust.scale import mad

from .data_stream import DataStream
from .operations import DataStreamOperation


class TrimStrategy(ABC):
    """
    Abstract base class describing a trim strategy.
    """

    def __init__(self, window_size: int = 10, start_time: float = 0.0, **kwargs):
        self.window_size = window_size
        self.start_time = start_time
        # Store any extra kwargs (like threshold, robust, etc.)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        pass  # pragma: no cover

    def apply(
        self, data_stream: DataStream, column_name: str, **kwargs: Any
    ) -> DataStream:
        """
        Template method that defines the trimming workflow.
        """
        # Check stationarity
        if not self._check_stationary(data_stream, column_name):
            return self._create_error_result(
                data_stream,
                kwargs,
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data.",
            )

        # Preprocess
        data = self._preprocess(data_stream, column_name)

        # Detect steady state (implemented by subclasses)
        steady_state_start_time = self._detection_method(data, column_name)

        # Handle result
        if steady_state_start_time is not None:
            return self._create_success_result(
                data_stream, column_name, steady_state_start_time, kwargs
            )
        else:
            return self._create_error_result(
                data_stream,
                kwargs,
                f"Steady-state start time could not be determined for column '{column_name}'.",
            )

    def _check_stationary(self, data_stream: DataStream, column_name: str) -> bool:
        """Check if the column is stationary."""
        stationary_result = data_stream.is_stationary(column_name)
        if isinstance(stationary_result, dict):
            return stationary_result.get(column_name) is True
        return False

    def _preprocess(self, data_stream: DataStream, column_name: str) -> pd.DataFrame:
        """Preprocess the data"""
        data = data_stream.data[
            data_stream.data["time"] >= self.start_time
        ].reset_index(drop=True)

        non_zero_index = data[data[column_name] > 0].index.min()
        if non_zero_index is not None and non_zero_index > 0:
            data = data.loc[non_zero_index:].reset_index(drop=True)

        return data

    def _create_success_result(
        self,
        data_stream: DataStream,
        column_name: str,
        steady_state_start_time: float,
        kwargs: dict,
    ) -> DataStream:
        """Create a successful trimmed result."""
        trimmed_df = data_stream.data[
            data_stream.data["time"] >= steady_state_start_time
        ][["time", column_name]].reset_index(drop=True)

        result = DataStream(trimmed_df, history=data_stream.history)
        kwargs["sss_start"] = steady_state_start_time
        return result

    def _create_error_result(
        self, data_stream: DataStream, kwargs: dict, message: str
    ) -> DataStream:
        """Create an empty result with error message."""
        empty_df = data_stream.data.iloc[0:0].copy()
        result = DataStream(empty_df, history=data_stream.history)
        result.message = message
        kwargs["message"] = message
        return result

    @abstractmethod
    def _detection_method(
        self, data: pd.DataFrame, column_name: str
    ) -> Optional[float]:
        """
        Detect the steady-state start time.
        Must be implemented by subclasses.

        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed data
        column_name : str
            Name of the column to analyze

        Returns
        -------
        float or None
            The time at which steady state begins, or None if not found.
        """
        pass


class QuantileTrimStrategy(TrimStrategy):
    """Trim based on sliding standard deviation criteria."""

    def __init__(
        self,
        window_size: int = 10,
        start_time: float = 0.0,
        robust: bool = True,
    ):
        super().__init__(
            window_size=window_size,
            start_time=start_time,
            robust=robust,
        )

    @property
    def method_name(self) -> str:
        return "std"

    def _detection_method(
        self,
        data: pd.DataFrame,
        column_name: str,
    ) -> Optional[float]:
        """
        Identify the earliest time point when the signal remains within ±1/2/3σ proportions.

        Parameters
        ----------
        data : DataFrame
            Subset of the original df (must include 'time' and signal column).
        column_name : str
        window_size : int
            Number of samples to evaluate the steady-state criteria.
        robust : bool
            If True, use median and MAD; else mean and std.

        Returns
        -------
        """

        window_size = self.window_size
        robust = self.robust

        time_filtered = data["time"].values
        signal_filtered = data[column_name].values

        if len(signal_filtered) < window_size:
            return None

        for i in range(len(signal_filtered) - window_size + 1):
            remaining_data = signal_filtered[i:]

            if robust:
                central_value = np.median(remaining_data)
                scale_value = mad(remaining_data)
            else:
                central_value = np.mean(remaining_data)
                scale_value = np.std(remaining_data)

            within_1 = np.mean(np.abs(remaining_data - central_value) <= scale_value)
            within_2 = np.mean(
                np.abs(remaining_data - central_value) <= 2 * scale_value
            )
            within_3 = np.mean(
                np.abs(remaining_data - central_value) <= 3 * scale_value
            )

            if within_1 >= 0.68 and within_2 >= 0.95 and within_3 >= 0.99:
                return time_filtered[i]

        return None


class NoiseThresholdTrimStrategy(TrimStrategy):
    """Trim using rolling standard deviation on normalized data."""

    def __init__(
        self,
        window_size: int = 10,
        start_time: float = 0.0,
        threshold: Optional[float] = None,
        robust: bool = True,
    ):
        super().__init__(
            window_size=window_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )

    def apply(self, data_stream: DataStream, column_name: str, **kwargs):
        # Stationarity check
        if not self._check_stationary(data_stream, column_name):
            return self._create_error_result(
                data_stream,
                kwargs,
                f"Column '{column_name}' is not stationary. "
                "Steady-state trimming requires stationary data.",
            )

        # Now threshold validation
        if self.threshold is None:
            return self._create_error_result(
                data_stream,
                kwargs,
                "Threshold must be specified for the 'threshold' trim strategy.",
            )

        return super().apply(data_stream, column_name, **kwargs)

    @property
    def method_name(self) -> str:
        return "threshold"

    def _detection_method(
        self,
        data: pd.DataFrame,
        column_name: str,
    ) -> Optional[float]:
        """
        Use rolling standard deviation on normalized data to detect steady-state.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Std threshold under which to mark steady-state.

        Returns
        -------
        float or None
        """
        window_size = self.window_size
        threshold = self.threshold

        normalized_data = DataStream.normalize_data(data.copy())
        time_series = normalized_data[["time", column_name]]

        if len(time_series) < window_size:
            return None

        rolling_std = time_series[column_name].rolling(window=window_size).std()
        common_idx = time_series.index.intersection(rolling_std.index)
        steady_state = time_series.loc[common_idx, "time"][
            rolling_std.loc[common_idx] < threshold
        ]

        if not steady_state.empty:
            return steady_state.iloc[0]

        return None


class RollingVarianceThresholdTrimStrategy(TrimStrategy):
    """Detect steady-state when rolling variance falls below threshold."""

    def __init__(
        self,
        window_size: int = 50,
        start_time: float = 0.0,
        robust: bool = True,
        threshold: Optional[float] = 0.1,
    ):
        super().__init__(
            window_size=window_size,
            start_time=start_time,
            robust=robust,
            threshold=threshold,
        )

    @property
    def method_name(self) -> str:
        return "rolling_variance"

    def _detection_method(
        self, data: pd.DataFrame, column_name: str
    ) -> Optional[float]:
        """
        Detect steady-state when rolling variance falls below a fraction of its mean.

        Parameters
        ----------
        data : DataFrame
        column_name : str
        window_size : int
        threshold : float
            Fraction of mean rolling std below which to consider steady-state.

        Returns
        -------
        float or None
            Time of first below-threshold variance, or None.
        """
        window_size = self.window_size
        threshold = self.threshold

        ts = data[["time", column_name]].dropna()
        time_values = ts["time"]
        signal_values = ts[column_name]
        rolling_variance = signal_values.rolling(window=window_size).std()
        threshold_val = rolling_variance.mean() * threshold
        steady_state_index = np.where(rolling_variance < threshold_val)[0]

        if len(steady_state_index) > 0:
            return time_values.iloc[steady_state_index[0]]

        return None


class MeanVariationTrimStrategy(TrimStrategy):
    """Trim using Statistical Steady State detection."""

    def __init__(
        self,
        *,
        max_lag_frac=None,
        verbosity=None,
        autocorr_sig_level=None,
        decor_multiplier=None,
        std_dev_frac=None,
        fudge_fac=None,
        smoothing_window_correction=None,
        final_smoothing_window=None,
    ):
        super().__init__(window_size=0, start_time=0.0)
        self.max_lag_frac = max_lag_frac
        self.verbosity = verbosity
        self.autocorr_sig_level = autocorr_sig_level
        self.decor_multiplier = decor_multiplier
        self.std_dev_frac = std_dev_frac
        self.fudge_fac = fudge_fac
        self.smoothing_window_correction = smoothing_window_correction
        self.final_smoothing_window = final_smoothing_window

    @property
    def method_name(self) -> str:
        return "sss_start"

    def _detection_method(
        self, data: pd.DataFrame, column_name: str
    ) -> Optional[float]:
        """Not used - SSS overrides apply() entirely."""
        raise NotImplementedError(
            "MeanVariationTrimStrategy overrides apply() and doesn't use _detection_method"
        )

    def apply(
        self, data_stream: DataStream, column_name: str, **kwargs: Any
    ) -> DataStream:
        """
        Identify and trim the signal to the start of the Statistical Steady State (SSS)

        Parameters
        ----------
        col : str
            The name of the column in `data_stream.data` to analyze for steady state.
        workflow : object
            A configuration/workflow object containing parameters:
            - `_max_lag_frac`: Fraction of data used for autocorrelation lag.
            - `_verbosity`: Integer controlling plot and print output levels.
            - `_autocorr_sig_level`: Significance level for the Z-test on lags.
            - `_decor_multiplier`: Multiplier for the calculated decorrelation length.
            - `_std_dev_frac`: Fraction of standard deviation used for tolerance.
            - `_fudge_fac`: Constant to prevent zero-tolerance in noiseless signals.
            - `_smoothing_window_correction`: Factor to adjust for rolling mean lag.
            - `_final_smoothing_window`: Window size for smoothing the metric curves.

        Returns
        -------
        DataStream
            A new DataStream object containing the DataFrame trimmed to the SSS start.
            Returns an empty DataFrame if no SSS is identified.
        """

        # Get the decorrelation length (in number of points)
        # Note: this approach assumes signal points are spaced equally in time
        n_pts = len(data_stream.data)
        max_lag = int(self.max_lag_frac * n_pts)  # max lag for autocorrelation

        acf_vals = ststls.acf(
            data_stream.data[column_name].dropna().values, nlags=max_lag
        )

        # plot the autocorrelation function
        if self.verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.stem(range(len(acf_vals)), acf_vals)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Autocorrelation Function")
            plt.grid()
            plt.show()
            plt.close()

        # Use rigorous statistical measure for decorrelation length
        z_critical = sts.norm.ppf(1 - self.autocorr_sig_level / 2)
        conf_interval = z_critical / np.sqrt(n_pts)
        significant_lags = np.where(np.abs(acf_vals[1:]) > conf_interval)[0]
        acf_sum = np.sum(np.abs(acf_vals[1:][significant_lags]))
        decor_length = int(np.ceil(1 + 2 * acf_sum))

        # Set smoothing window as multiple of decorrelation length, but not more than max_lag
        decor_index = min(int(self.decor_multiplier * decor_length), max_lag)

        if self.verbosity > 0:
            print(
                f"stats decorrelation length {decor_length} gives smoothing window of {decor_index} points."
            )

        # Smooth signal with rolling mean over window size based on decorrelation length
        rolling_window = max(3, decor_index)  # at least 3 points in window
        col_smoothed = (
            data_stream.data[column_name].rolling(window=rolling_window).mean()
        )  # get smoothed column as Series
        col_sm_flld = col_smoothed.bfill()  # fill initial NaNs with first valid value
        # create new DataFrame with time and smoothed flux
        df_smoothed = pd.DataFrame(
            {"time": data_stream.data["time"], column_name: col_sm_flld}
        )

        # Compute std dev of original signal from current location till end of signal
        std_dev_till_end = np.empty((n_pts,), dtype=float)
        for i in range(n_pts):
            std_dev_till_end[i] = np.std(data_stream.data[column_name].iloc[i:])
        # turn this into a pandas series with same index as col_smoothed
        std_dev_till_end_series = pd.Series(
            std_dev_till_end, index=data_stream.data.index
        )
        # Smooth this std dev to avoid it going to zero at end of signal
        std_dev_smoothed = std_dev_till_end_series.rolling(
            window=self.final_smoothing_window
        ).mean()
        # Fill initial NaNs with the first valid smoothed std dev value
        std_dev_sm_flld = std_dev_smoothed.bfill()

        df_std_dev = pd.DataFrame(
            {
                "time": data_stream.data["time"],
                f"{column_name}_std_till_end": std_dev_sm_flld,
            }
        )

        # start time of smoothed signal
        smoothed_start_time = df_smoothed["time"].iloc[rolling_window - 1]

        # plot smoothed signal and related quantities
        if self.verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(
                data_stream.data["time"],
                data_stream.data[column_name],
                label="Original Signal",
                alpha=0.5,
            )
            plt.plot(
                df_smoothed["time"],
                df_smoothed[column_name],
                label="Smoothed Signal",
                color="orange",
            )
            plt.plot(
                df_std_dev["time"],
                df_std_dev[column_name + "_std_till_end"],
                label="Smoothed Std Dev Till End",
                color="green",
            )
            plt.axvline(
                x=smoothed_start_time,
                color="g",
                linestyle="--",
                label="First smoothed point",
            )
            plt.xlabel("Time")
            plt.ylabel(column_name)
            plt.title("Original and Smoothed Signal")
            plt.legend()
            plt.grid()
            plt.show()
            plt.close()

        if self.verbosity > 0:
            print("Getting start of SSS based on smoothed signal:")

        # Get start of SSS based on where the value of the flux in the smoothed signal
        # is close to the mean of the remaining signal.

        # At each location, compute the mean of the remaining smoothed signal
        n_pts_smoothed = len(df_smoothed)
        mean_vals = np.empty((n_pts_smoothed,), dtype=float)

        for i in range(n_pts_smoothed):
            mean_vals[i] = np.mean(df_smoothed[column_name].iloc[i:])

        # Check where the current value of the smoothed signal is within tol_fac of the mean of the remaining signal
        deviation_arr = np.abs(df_smoothed[column_name] - mean_vals)

        # smooth this so the deviation does not go to zero at end of signal by construction
        # turn this into a pandas series with same index as col_smoothed
        deviation_series = pd.Series(deviation_arr, index=data_stream.data.index)
        # Smooth this std dev to avoid it going to zero at end of signal
        deviation_smoothed = deviation_series.rolling(
            window=self.final_smoothing_window
        ).mean()
        # Fill initial NaNs with the first valid smoothed std dev value
        deviation_sm_flld = deviation_smoothed.bfill()
        # Build a dataframe for the deviation
        deviation = pd.DataFrame(
            {
                "time": data_stream.data["time"],
                column_name + "_deviation": deviation_sm_flld,
            }
        )

        # Compute tolerance on variation in the mean of the smoothed signal as
        # stdv_frac * (std dev till end + a fudge factor * mean value at start of smoothed signal)
        # fudge factor is for in case there is no noise (and to guard against the tolerance
        # factor going to zero when std dev gets very small at end of signal)
        tol_fac = self.std_dev_frac * (
            df_std_dev[column_name + "_std_till_end"]
            + self.fudge_fac * abs(mean_vals[0])
        )
        tolerance = tol_fac * np.abs(mean_vals)

        within_tolerance_all = deviation[column_name + "_deviation"] <= tolerance
        # Only consider points after the smoothed signal has started
        within_tolerance = within_tolerance_all & (
            df_smoothed["time"] >= smoothed_start_time
        )
        # First index where we are within tolerance
        sss_index = np.where(within_tolerance)[0]

        # See if there is a segment where ALL remaining points are within tolerance
        crit_met_index = None
        if len(sss_index) > 0:
            # find the segment where ALL remaining points are within tolerance
            for idx in sss_index:
                if np.all(within_tolerance[idx:]):
                    crit_met_index = idx
                    break

        if crit_met_index is not None:  # We have a SSS segment
            # Time where criterion has been met
            criterion_time = df_smoothed["time"].iloc[crit_met_index]
            # Take into account that the signal at the point where the criterion has been met is a result
            # of averaging over the rolling window. So set the start of SSS near the start of the rolling window
            # but not all the way at the beginning of the rolling window as there is usually still some transient.
            true_sss_start_index = max(
                0,
                int(crit_met_index - self.smoothing_window_correction * rolling_window),
            )
            sss_start_time = df_smoothed["time"].iloc[true_sss_start_index]

            if self.verbosity > 0:
                print(f"Index where criterion is met: {crit_met_index}")
                print(f"Rolling window: {rolling_window}")
                print(f"time where criterion is met: {criterion_time}")
                print(
                    f"time at start of SSS (adjusted for rolling window): {sss_start_time}"
                )

            # Plot deviation and tolerance vs. time
            if self.verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    df_smoothed["time"],
                    deviation[column_name + "_deviation"],
                    label="Deviation",
                    color="blue",
                )
                plt.plot(
                    df_smoothed["time"],
                    tolerance,
                    label="Tolerance",
                    color="orange",
                )
                plt.axvline(
                    x=criterion_time,
                    color="g",
                    linestyle="--",
                    label="Small Change Criterion Met",
                )
                plt.axvline(
                    x=sss_start_time, color="r", linestyle="--", label="Start SSS"
                )
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title("Deviation and Tolerance vs. Time")
                plt.legend()
                plt.grid()
                plt.show()
                plt.close()

            # Trim the original data frame to start at this location minus the smoothing window
            trimmed_df = data_stream.data[data_stream.data["time"] >= sss_start_time]
            # Reset the index so it starts at 0
            trimmed_df = trimmed_df.reset_index(drop=True)
            # Create new data stream from trimmed data frame
            trimmed_stream = DataStream(trimmed_df)

        else:
            if self.verbosity > 0:
                print("No SSS found based on behavior of mean of smoothed signal.")
            trimmed_stream = pd.DataFrame(
                columns=["time", "flux"]
            )  # Create empty DataFrame with same columns as original

            # Plot deviation and tolerance vs. time
            if self.verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    df_smoothed["time"],
                    deviation[column_name + "_deviation"],
                    label="Deviation",
                    color="blue",
                )
                plt.plot(
                    df_smoothed["time"],
                    tolerance,
                    label="Tolerance",
                    color="orange",
                )
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.title("Deviation and Tolerance vs. Time")
                plt.legend()
                plt.grid()
                plt.show()
                plt.close()

        return trimmed_stream


StandardDeviationTrimStrategy = QuantileTrimStrategy
ThresholdTrimStrategy = NoiseThresholdTrimStrategy
RollingVarianceTrimStrategy = RollingVarianceThresholdTrimStrategy
SSSStartTrimStrategy = MeanVariationTrimStrategy


class TrimDataStreamOperation(DataStreamOperation):
    """
    Operation that applies a TrimStrategy to a DataStream.
    """

    def __init__(self, strategy: TrimStrategy, operation_name: str = "trim") -> None:
        super().__init__(
            operation_name=operation_name,
            strategy=type(strategy).__name__,
        )
        self._strategy = strategy

    @property
    def strategy(self) -> TrimStrategy:
        return self._strategy

    # Override to skip base history
    def __call__(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        return self._apply(data_stream, **kwargs)

    def _apply(self, data_stream: DataStream, **kwargs: Any) -> DataStream:
        """Apply the strategy and build history."""
        column_name = kwargs.get("column_name")

        # Build base parameters that all strategies share
        options = {
            "column_name": column_name,
            "window_size": self._strategy.window_size,
            "start_time": self._strategy.start_time,
            "method": self._strategy.method_name,
        }

        # Add strategy-specific parameters
        if hasattr(self._strategy, "robust"):
            options["robust"] = self._strategy.robust
        if hasattr(self._strategy, "threshold"):
            options["threshold"] = self._strategy.threshold

        # Apply strategy
        result = self._strategy.apply(data_stream, **kwargs)

        # Strategy may attach an error message to the result
        if hasattr(result, "message"):
            options["message"] = result.message

        # Build history as a dictionary
        result._history = [
            {"operation": "is_stationary", "options": {"columns": column_name}},
            {"operation": "trim", "options": options},
        ]

        return result
