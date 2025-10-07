import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.tsa.stattools as ststls

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
        0: very few print statements or plots
        1: more print statements
        2: also show plots of intermediate steps
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

    """

    def __init__(self,
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
                 smoothing_window_correction=0.8
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

        if self._operate_safe: # Not stationary or no steady state and we want to operate safely
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

            return results_dict
        
        else: # not stationary or no steady state, but we want to get some result back

            # Return ad-hoc mean based on last 33% of data and arbitrarily large uncertainties/confidence interval
            results_dict = {}
            results_dict[col] = {}

            # Get all data that is later than start_time
            df_past_start = data_stream.df[data_stream.df['time'] >= start_time]

            # Get the data
            column_data = df_past_start[col].dropna()

            # Compute index for 2/3rds of the data set
            n_pts = len(column_data)
            n_66pc = (n_pts*2)//3
            
            # Get ad hoc statistics
            mean_val = np.mean(column_data[n_66pc:])
            uncertainty_val = mean_val # Arbitrary 100% uncertainty
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
        n_pts_orig = len(ds_wrk.df)

        # Check if data stream is stationary
        # TODO: spin this whole operation to determine stationarity off into
        # a separate function (part of base DataStream class?)
        stationary = ds_wrk.is_stationary([col])[col] # is_stationary() returns dictionary. The value for key qoi tells us if it is stationary

        n_pts = len(ds_wrk.df)
        while not stationary and not self._operate_safe and n_pts > self._n_pts_min and n_pts > self._n_pts_frac_min*n_pts_orig:
            # See if we get a stationary stream if we drop some initial fraction of the data
            n_drop = int(n_pts*self._drop_fraction)
            df_shortened = ds_wrk.df.iloc[n_drop:]
            ds_wrk = DataStream(df_shortened)
            n_pts = len(ds_wrk.df)
            n_dropped = n_pts_orig - n_pts
            stationary = ds_wrk.is_stationary([col])[col]
            if self._verbosity > 0:
                if stationary:
                    print(f"Data stream was not stationary, but is stationary after dropping first {n_dropped} points.")
                else:
                    print(f"Data stream is not stationary, even after dropping first {n_dropped} points.")

        if stationary:

            # Trim the data stream to find statistical steady state
            # TODO: spin this off into a separate function (part of base DataStream class?)

            # Get the decorrelation length (in number of points)
            # Note: this approach assumes signal points are spaced equally in time
            n_pts = len(ds_wrk.df)
            max_lag = int(self._max_lag_frac*n_pts) # max lag for autocorrelation

            acf_vals = ststls.acf(ds_wrk.df[col].dropna().values, nlags=max_lag)

            # plot the autocorrelation function
            if self._verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.stem(range(len(acf_vals)), acf_vals)
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.title('Autocorrelation Function')
                plt.grid()
                plt.show()
                plt.close()

            # Use rigorous statistical measure for decorrelation length
            z_critical = sts.norm.ppf(1 - self._autocorr_sig_level / 2)
            conf_interval = z_critical / np.sqrt(n_pts)
            significant_lags = np.where(np.abs(acf_vals[1:]) > conf_interval)[0]
            acf_sum = np.sum(np.abs(acf_vals[1:][significant_lags]))
            decor_length = int(np.ceil(1 + 2 * acf_sum))

            # Set smoothing window as multiple of decorrelation length, but not more than max_lag
            decor_index = min(int(self._decor_multiplier*decor_length), max_lag)

            if self._verbosity > 0:
                print(f"stats decorrelation length {decor_length} gives smoothing window of {decor_index} points.")

            # Smooth signal with rolling mean over window size based on decorrelation length
            rolling_window = max(3,decor_index) # at least 3 points in window
            col_smoothed = ds_wrk.df[col].rolling(window=rolling_window).mean() # get smoothed column as Series

            # Compute std dev of original signal from current location till end of signal
            std_dev_till_end = np.empty((n_pts,),dtype=float)
            for i in range(n_pts):
                std_dev_till_end[i] = np.std(ds_wrk.df[col].iloc[i:])
            # turn this into a pandas series
            std_dev_till_end_series = pd.Series(std_dev_till_end)
            # Smooth this std dev to avoid it going to zero at end of signal
            std_dev_smoothed = std_dev_till_end_series.rolling(window=10).mean()
            # Fill initial NaNs with the first valid smoothed std dev value
            std_dev_sm_flld = std_dev_smoothed.fillna(method='bfill')


            # create new DataFrame with time, smoothed flux and std dev till end of signal
            df_smoothed = pd.DataFrame({'time': ds_wrk.df['time'], col: col_smoothed, col+'_std_till_end': std_dev_sm_flld}) 

            # plot smoothed signal
            if self._verbosity > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(ds_wrk.df['time'], ds_wrk.df[col], label='Original Signal', alpha=0.5)
                plt.plot(df_smoothed['time'], df_smoothed[col], label='Smoothed Signal', color='orange')
                plt.xlabel('Time')
                plt.ylabel(col)
                plt.title('Original and Smoothed Signal')
                plt.legend()
                plt.grid()
                plt.show()
                plt.close()

            if self._verbosity > 0:
                print("Getting start of SSS based on smoothed signal:")


            # Get start of SSS based on where the value of the flux in the smoothed signal
            # is within tol_fac of the mean of the remaining signal, 
            # where tol_fac = factor * the rolling std dev of the stationary signal

            # At each location, compute the mean of the remaining signal
            n_pts_smoothed = len(df_smoothed)
            mean_vals = np.empty((n_pts_smoothed,),dtype=float)
            # stdv_vals = np.empty((n_pts_smoothed,),dtype=float)
            for i in range(n_pts_smoothed):
                mean_vals[i] = np.mean(df_smoothed[col].iloc[i:])
                # stdv_vals[i] = np.std(df_smoothed[col].iloc[i:])
            # Check where the current value of the smoothed signal is within tol_fac of the mean of the remaining signal
            deviation = np.abs(df_smoothed[col] - mean_vals)
            # Compute tolerance on variation in the mean of the smoothed signal as
            # stdv_frac * std dev till end + a fudge factor * mean value at start of smoothed signal
            # in case there is no noise (and to guard against the tolerance
            # factor going to zero when std dev goes to 0 at end of signal)
            tol_fac = self._std_dev_frac * (df_smoothed[col+'_std_till_end'] + self._fudge_fac*abs(mean_vals[0]))
            tolerance = tol_fac * np.abs(mean_vals)

            within_tolerance = deviation <= tolerance
            sss_index = np.where(within_tolerance)[0]

            if len(sss_index) > 0:
                # find the segment where ALL remaining points are within tolerance
                for idx in sss_index:
                    if np.all(within_tolerance[idx:]):
                        crit_met_index = idx
                        break

                # Time where criterion has been met
                criterion_time = df_smoothed['time'].iloc[crit_met_index]
                # Take into account that the signal at the point where the criterion has been met is a result
                # of averaging over the rolling window. So set the start of SSS near the start of the rolling window
                # but not all the way at the beginning of the rolling window as there is usually still some transient.
                true_sss_start_index = max(0, int(crit_met_index - self._smoothing_window_correction*rolling_window))
                sss_start_time = df_smoothed['time'].iloc[true_sss_start_index]

                if self._verbosity > 0:
                    print(f"Index where criterion is met: {crit_met_index}")
                    print(f"Rolling window: {rolling_window}")
                    print(f"time where criterion is met: {criterion_time}")
                    print(f"time at start of SSS (adjusted for rolling window): {sss_start_time}")

                # Plot deviation and tolerance vs. time
                if self._verbosity > 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_smoothed['time'], deviation, label='Deviation', color='blue')
                    plt.plot(df_smoothed['time'], tolerance, label='Tolerance', color='orange')
                    plt.axvline(x=criterion_time, color='g', linestyle='--', label="Small Change Criterion Met")
                    plt.axvline(x=sss_start_time, color='r', linestyle='--', label="Start SSS")
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.title('Deviation and Tolerance vs. Time')
                    plt.legend()
                    plt.grid()
                    plt.show()
                    plt.close()

                # Trim the original data frame to start at this location minus the smoothing window
                trimmed_df = ds_wrk.df[ds_wrk.df['time'] >= sss_start_time]
                # Reset the index so it starts at 0
                trimmed_df = trimmed_df.reset_index(drop=True)
                # Create new data stream from trimmed data frame
                trimmed_stream = DataStream(trimmed_df)
            else:
                print(f"No SSS found based on {tol_fac*100}% criterion")
                trimmed_stream = pd.DataFrame(columns=['time', 'flux']) # Create empty DataFrame with same columns as original


            # Check that a steady state was found
            if len(trimmed_stream) > 1:
                
                if self._verbosity > 0:
                    print("Trimmed data frame:")
                    print(trimmed_stream.df.head())

                # Start time of statistical steady state
                sss_start = trimmed_stream.df['time'][0]

                # Get statistics (with window selected by decorrelation length)
                trimmed_stats = trimmed_stream.compute_statistics(column_name=col)

                # Add flag for the results for this qoi that all is normal
                trimmed_stats[col]["sss_start"] = sss_start
                trimmed_stats[col]["metadata"] = {}
                trimmed_stats[col]["metadata"]["status"] = "Regular"
                trimmed_stats[col]["metadata"]["mitigation"] = "None"

            else: # No statistical steady state
                if self._verbosity > 0:
                    print("No statistical steady state found after trimming.")
                # Alternative processing
                trimmed_stats = self.process_irregular_stream(data_stream_orig, col, start_time=start_time)

        else: # Not stationary
            if self._verbosity > 0:
                print("Data stream is not stationary.")
            # Alternative processing
            trimmed_stats = self.process_irregular_stream(data_stream_orig, col, start_time=start_time)

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

        ax.plot(my_df["time"],my_df[col], label="Signal")

        ax.set_xlabel("time", size=12)
        ax.set_ylabel(col, size=12)
        if label:
            ax.set_title(label, size=14)
        # Set the font size for the axis tick labels
        ax.tick_params(axis='both', labelsize=11)

        # add the start of steady state and the mean (if provided)
        if stats:
            my_mean = stats[col]["mean"]
            my_cl = stats[col]["confidence_interval"]
            my_sss_start = stats[col]["sss_start"]
            plt.axvline(x=my_sss_start, color='r', linestyle='--', label="Start SSS")

            sss_time = [my_sss_start,my_df.iloc[-1]["time"]]
            mean_level = [my_mean, my_mean]
            upper_conf_level = [my_cl[1],my_cl[1]]
            lower_conf_level = [my_cl[0],my_cl[0]]
            ax.plot(sss_time,mean_level, color='green', linestyle='-', label="Mean")
            ax.plot(sss_time,upper_conf_level, color='green', linestyle='--', label="95% Conf. Int.")
            ax.plot(sss_time,lower_conf_level, color='green', linestyle='--')
        ax.legend(fontsize=12)

        # show and close the figure
        plt.show(fig)
        plt.close(fig)
