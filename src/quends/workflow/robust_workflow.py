import matplotlib.pyplot as plt
import numpy as np

# QUENDS libraries
from ..base.data_stream import DataStream


class RobustWorkflow:
    """
    Set of functions to analyze DataStreams in a robust way.

    This class can handle data streams with a lot of noise and where stationarity or the start
    of steady statistical state (SSS) can be hard to assess. It uses base DataStream methods for statistical
    analysis but adds alternative tools for stationarity assessment and start of SSS detection.

    Note: this class assumes the time points in the data stream are equally spaced in time.

    Core features include:
    - Stationarity assessment that progressively shortens the DataStream to see if the tail
      end of the DataStream is stationary.
    - Start of SSS detection that uses a robust approach based on the smoothed mean of the DataStream.
    - Methods that return "ball park" statistics if the DataStream is not stationary,
      or if there is no SSS segment found.

    Attributes
    ----------
    _drop_fraction: float, fraction of data to drop from the start of the DataStream to see if the shortened
        DataStream is stationary.
    _operate_safe : bool
        If True: process data streams in a safe way insisting on stationarity and a segment
        that is clearly in SSS
        If False: try to get some results even if the data stream is not stationary or there is no
        SSS segment found.
    _verbosity: int, level of verbosity for print statements and plots.
        0. : very few print statements or plots
        > 0: more print statements
        > 1: also show plots of intermediate steps
    _drop_fraction: float, fraction of data to drop from the start of the DataStream to see if the shortened
        DataStream is stationary.
    _n_pts_min: int, minimum number of points to keep in the DataStream when shortening it to check for stationarity.
    _n_pts_frac_min: float, minimum fraction of the original number of points to keep in the DataStream when shortening it
        to check for stationarity.
    _max_lag_frac: float, maximum lag (as a fraction of the number of points in the DataStream) to use when computing
        the autocorrelation function to determine the decorrelation length.
    _autocorr_sig_level: float, significance level to use when determining the decorrelation length from the autocorrelation
        function.
    _decor_multiplier: float, multiplier to apply to the decorrelation length to get the smoothing window size.
    _std_dev_frac: float, fraction of the std dev of the stationary signal to use as tolerance when determining the start
        of SSS.
    _fudge_fac: float, fudge factor to multiply the initial mean of the smoothed signal with before adding it to the std dev
        used to compute the tolerance for determining the start of SSS.
    _smoothing_window_correction: float, correction factor to apply to the smoothing window size when determining the start of SSS.
    _final_smoothing_window: int, smoothing window used to avoid quantities going to zero at end of signal.

    """

    def __init__(
        self,
        operate_safe=True,
        verbosity=0,
        drop_fraction=0.25,
        n_pts_min=100,
        n_pts_frac_min=0.2,
        max_lag_frac=0.5,
        autocorr_sig_level=0.05,
        decor_multiplier=4.0,
        std_dev_frac=0.1,
        fudge_fac=0.1,
        smoothing_window_correction=0.8,
        final_smoothing_window=10,
    ):
        """
        Initialize a workflow and its hyperparameters

        Parameters
        ----------
        operate_safe: bool, optional
            If True: process data streams in a safe way insisting on stationarity and a segment
            that is clearly in SSS
            If False: try to get some results even if the data stream is not stationary or there is no
            SSS segment found. Default is True.
        verbosity: int, optional
            Level of verbosity for print statements and plots. Default is 0 (very few print statements or
            plots). Higher values give more print statements and plots.
        drop_fraction: float, optional
            Fraction of data to drop from the start of the DataStream to see if the shortened
            DataStream is stationary. Default is 0.25 (25% of data).
        n_pts_min: int, optional
            Minimum number of points to keep in the DataStream when shortening it to check for stationarity.
            Default is 100 points.
        n_pts_frac_min: float, optional
            Minimum fraction of the original number of points to keep in the DataStream when shortening it
            to check for stationarity. Default is 0.2 (20% of original number of points).
        max_lag_frac: float, optional
            Maximum lag (as a fraction of the number of points in the DataStream) to use when computing
            the autocorrelation function to determine the decorrelation length. Default is 0.5.
        autocorr_sig_level: float, optional
            Significance level to use when determining the decorrelation length from the autocorrelation
            function. Default is 0.05 (5% significance level).
        decor_multiplier: float, optional
            Multiplier to apply to the decorrelation length to get the smoothing window size. Default is 4.0
        std_dev_frac: float, optional
            Fraction of the std dev of the stationary signal to use as tolerance when determining the start
            of SSS. Default is 0.1 (10% of std dev).
        fudge_fac: float, optional
            Fudge factor to multiply the initial mean of the smoothed signal with before adding it to the std dev
            used to compute the tolerance for determining the start of SSS. This is to guard against
            the tolerance going to zero when the std dev goes to zero at the end of the signal.
            Default is 0.1 (10% of mean value at start of smoothed signal).
        smoothing_window_correction: float, optional
            Correction factor to apply to the smoothing window size when determining the start of SSS.
            This is to account for the fact that the smoothed signal at a given time point is
            the result of averaging over the smoothing window. So the SSS can be seen as starting
            before the point where the tolerance is met.
            Default is 0.8 (80% of smoothing window size).
        final_smoothing_window: int, optional
            Smoothing window used to avoid quantities going to zero at end of signal. Default is 10 points.
        """
        self._operate_safe = operate_safe
        self._verbosity = verbosity
        self._drop_fraction = drop_fraction
        self._n_pts_min = n_pts_min
        self._n_pts_frac_min = n_pts_frac_min
        self._max_lag_frac = max_lag_frac
        self._autocorr_sig_level = autocorr_sig_level
        self._decor_multiplier = decor_multiplier
        self._std_dev_frac = std_dev_frac
        self._fudge_fac = fudge_fac
        self._smoothing_window_correction = smoothing_window_correction
        self._final_smoothing_window = final_smoothing_window

    def process_irregular_stream(self, data_stream, col, start_time=0.0):
        """
        Process a data stream that is not stationary or has no steady state segment

        Parameters
        ----------
        data_stream: DataStream
            The data stream to process.
        col: str
            The column name of the quantity of interest in the data stream.
        start_time: float, optional
            The time after which to consider data for processing. Default is 0.0.

        Returns
        -------
        results_dict: dict
            Dictionary with results for the quantity of interest.

        """

        if (
            self._operate_safe
        ):  # Not stationary or no steady state and we want to operate safely
            # Do not process and return non-stationary flag
            results_dict = {}
            results_dict[col] = {}
            results_dict[col]["mean"] = np.nan
            results_dict[col]["mean_uncertainty"] = np.nan
            results_dict[col]["confidence_interval"] = (np.nan, np.nan)
            results_dict[col]["sss_start"] = np.nan
            results_dict[col]["metadata"] = {}
            results_dict[col]["metadata"]["status"] = "NoStatSteadyState"
            results_dict[col]["metadata"]["mitigation"] = "Drop"
            # Add the start time info used
            results_dict[col]["start_time"] = start_time

            return results_dict

        else:  # not stationary or no steady state, but we want to get some result back

            # Return ad-hoc mean based on last 33% of data and arbitrarily large uncertainties/confidence interval
            results_dict = {}
            results_dict[col] = {}

            # Get all data that is later than start_time
            df_past_start = data_stream.df[data_stream.df["time"] >= start_time]

            # Get the data
            column_data = df_past_start[col].dropna()

            # Compute index for 2/3rds of the data set
            n_pts = len(column_data)
            n_66pc = (n_pts * 2) // 3

            # Get ad hoc statistics
            mean_val = np.mean(column_data[n_66pc:])
            uncertainty_val = mean_val  # Arbitrary 100% uncertainty
            # For confidence interval, assume the true value is somewhere between 0 and twice the mean.
            ci_lower = mean_val - uncertainty_val
            ci_upper = mean_val + uncertainty_val

            # Store results in dictionary and return
            results_dict[col]["mean"] = mean_val
            results_dict[col]["mean_uncertainty"] = uncertainty_val
            results_dict[col]["confidence_interval"] = (ci_lower, ci_upper)
            results_dict[col]["sss_start"] = df_past_start.iloc[n_66pc]["time"]
            results_dict[col]["metadata"] = {}
            results_dict[col]["metadata"]["status"] = "NoStatSteadyState"
            results_dict[col]["metadata"]["mitigation"] = "AdHoc"
            # Add the start time info used
            results_dict[col]["start_time"] = start_time

            return results_dict

    def process_data_steam(self, data_stream_orig, col, start_time=0.0):
        """
        Process data_stream and handle exceptions gracefully.
        Return mean value and its statistics


        TODO
        * look at number of effective samples we have. Could be low. Allow user to
        override this if they want minimum # of samples for analysis.

        Parameters
        ----------
        data_stream: DataStream
            The data stream to process.
        col: str
            The column name of the quantity of interest in the data stream.
        start_time: float, optional
            The time after which to consider data for processing. Default is 0.0.

        Returns
        -------
        results_dict: dict
            Dictionary with results for the quantity of interest.
        """

        # Work on a copy of the data stream
        ds_wrk = DataStream(data_stream_orig.df.copy())

        # Get all data that is later than start_time
        ds_wrk.df = ds_wrk.df[ds_wrk.df["time"] >= start_time]
        # Get number of points that we are working with
        n_pts_orig = len(ds_wrk.df)

        if self._verbosity > 0:
            print(f"Original size of data stream: {len(data_stream_orig.df)} points.")
            print(f"After enforcing start time there are {n_pts_orig} points left.")

        # Check if data stream is stationary

        # Check if it isn't stationary, if not drop fraction of data points
        ds_wrk, stationary = ds_wrk.make_stationary(col, n_pts_orig, self)

        if stationary:

            # detect and trim data stream to the start of statistcal steady state
            trimmed_stream = ds_wrk.trim_sss_start(col, self)

            # Check that a steady state was found
            if len(trimmed_stream) > 1:

                # if self._verbosity > 0:
                #     print("Trimmed data frame:")
                #     print(trimmed_stream.df.head())

                # Start time of statistical steady state
                sss_start = trimmed_stream.df["time"][0]

                # Get statistics (with window selected by decorrelation length)
                trimmed_stats = trimmed_stream.compute_statistics(column_name=col)

                # Add flag for the results for this qoi that all is normal
                trimmed_stats[col]["sss_start"] = sss_start
                trimmed_stats[col]["metadata"] = {}
                trimmed_stats[col]["metadata"]["status"] = "Regular"
                trimmed_stats[col]["metadata"]["mitigation"] = "None"
                # Add the start time info used
                trimmed_stats[col]["start_time"] = start_time

            else:  # No statistical steady state
                if self._verbosity > 0:
                    print("No statistical steady state found after trimming.")
                # Alternative processing
                trimmed_stats = self.process_irregular_stream(
                    data_stream_orig, col, start_time=start_time
                )

        else:  # Not stationary
            if self._verbosity > 0:
                print("Data stream is not stationary.")
            # Alternative processing
            trimmed_stats = self.process_irregular_stream(
                data_stream_orig, col, start_time=start_time
            )

        # Return the statistics dictionary
        return trimmed_stats

    # New function to plot signal with basic stats
    def plot_signal_basic_stats(self, data_stream, col, stats=None, label=None):
        """
        NOTE: make this part of visualization class?

        Parameters
        ----------
        data_stream: DataStream
            The data stream to plot
        col: str
            The column name of the quantity to plot in the data stream.
        stats: dict, optional
            Dictionary with statistics returned by process_data_steam(). Default is None.
        label: str, optional
            Label to use in title of graph. Default is None.

        Returns
        -------
        shows a plot of the signal with mean, confidence interval and start of SSS (if stats provided)
        """

        my_df = data_stream.df

        fig, ax = plt.subplots(figsize=(10, 6))

        (signal_line,) = ax.plot(my_df["time"], my_df[col], label="Signal")
        signal_color = signal_line.get_color()

        ax.set_xlabel("time", size=12)
        ax.set_ylabel(col, size=12)
        if label:
            ax.set_title(label, size=14)
        # Set the font size for the axis tick labels
        ax.tick_params(axis="both", labelsize=11)

        # add the start of steady state and the mean (if provided)
        if stats:
            # If start_time > 0, show it on graph
            if stats[col]["start_time"] > 0:
                ax.axvline(
                    x=stats[col]["start_time"],
                    color=signal_color,
                    linestyle="--",
                    label="Start Time",
                )

            # Add other statistics
            my_mean = stats[col]["mean"]
            my_cl = stats[col]["confidence_interval"]
            my_sss_start = stats[col]["sss_start"]
            plt.axvline(x=my_sss_start, color="r", linestyle="--", label="Start SSS")

            sss_time = [my_sss_start, my_df.iloc[-1]["time"]]
            mean_level = [my_mean, my_mean]
            upper_conf_level = [my_cl[1], my_cl[1]]
            lower_conf_level = [my_cl[0], my_cl[0]]
            ax.plot(sss_time, mean_level, color="green", linestyle="-", label="Mean")
            ax.plot(
                sss_time,
                upper_conf_level,
                color="green",
                linestyle="--",
                label="95% Conf. Int.",
            )
            ax.plot(sss_time, lower_conf_level, color="green", linestyle="--")

        ax.legend(fontsize=12)

        # show and close the figure
        plt.show(fig)
        plt.close(fig)
