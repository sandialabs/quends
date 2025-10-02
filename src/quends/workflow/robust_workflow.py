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
    of steady statistical state (SSS) can be hard to assess. It uses DataStream methods for statistical 
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
    _drop_fraction: fraction of data to drop from the start of the DataStream to see if the shortened
        DataStream is stationary.

    """

    def __init__(self, operate_safe=True, verbosity=0, drop_fraction=0.25):
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
        """
        self._operate_safe = operate_safe
        self._verbosity = verbosity
        self._drop_fraction = drop_fraction


    def process_irregular_stream(self, data_stream, col, start_time=0.0):
        """
        Process a data stream that is not stationary or has no steady state segment

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
        * look at number of effective samples we have. Could be low. Allow user to override this if they want minimum # of samples for analysis. 
        """

        # Work on a copy of the data stream
        ds_wrk = DataStream(data_stream_orig.df.copy())
        n_pts_orig = len(ds_wrk.df)

        # Check if data stream is stationary
        # TODO: spin this off into a separate function (part of base DataStream class?)
        stationary = ds_wrk.is_stationary([col])[col] # is_stationary() returns dictionary. The value for key qoi tells us if it is stationary

        n_pts = len(ds_wrk.df)
        # TODO: input these hardwired values as class attributes
        while not stationary and not self._operate_safe and n_pts > 100 and n_pts > 0.2*n_pts_orig:
            # See if we get a stationary stream if we drop some initial fraction of the data
            n_pts = len(ds_wrk.df)
            n_drop = int(n_pts*self._drop_fraction)
            df_shortened = ds_wrk.df.iloc[n_drop:]
            ds_wrk = DataStream(df_shortened)
            stationary = ds_wrk.is_stationary([col])[col]
            if self._verbosity > 0:
                # TODO: track total number of points dropped
                if stationary:
                    print(f"Data stream was not stationary, but is stationary after dropping first {n_drop} points.")
                else:
                    print(f"Data stream is not stationary, even after dropping first {n_drop} points.")

        if stationary:

            # Trim the data stream to find statistical steady state
            # TODO: spin this off into a separate function (part of base DataStream class?)

            # first get the std dev of the stationary signal (will be needed later)
            # std_dev_stat_signal = np.std(ds_wrk.df[col])

            # Get the decorrelation length (in number of points)
            # Note: this approach assumes signal points are spaced equally in time
            n_pts = len(ds_wrk.df)
            # TODO: input this value of 0.5
            max_lag = int(0.5*n_pts) # max lag for autocorrelation is half the data length

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
            alpha = 0.05  # 5% significance level
            z_critical = sts.norm.ppf(1 - alpha / 2)
            conf_interval = z_critical / np.sqrt(n_pts)
            significant_lags = np.where(np.abs(acf_vals[1:]) > conf_interval)[0]
            acf_sum = np.sum(np.abs(acf_vals[1:][significant_lags]))

            decor_length = int(np.ceil(1 + 2 * acf_sum))
            decor_index = min(4*decor_length, max_lag) # Smooth over 4 times the decorrelation length

            if self._verbosity > 0:
                print(f"stats decorrelation length {decor_length} gives smoothing window of {decor_index} points.")

            # Smooth signal with rolling mean over window size based on decorrelation length
            rolling_window = max(3,decor_index) # at least 3 points in window
            col_smoothed = ds_wrk.df[col].rolling(window=rolling_window).mean() # get smoothed column as Series
            col_rol_std = ds_wrk.df[col].rolling(window=rolling_window).std() # get rolling window based std dev as Series
            df_smoothed = pd.DataFrame({'time': ds_wrk.df['time'], col: col_smoothed, col+'_std': col_rol_std}) # create new DataFrame with time, smoothed flux and rolling std dev

            # Another experiment, compute std dev of original signal from current location till end of signal
            std_dev_till_end = np.empty((n_pts,),dtype=float)
            for i in range(n_pts):
                std_dev_till_end[i] = np.std(ds_wrk.df[col].iloc[i:])
            df_smoothed[col+'_std_till_end'] = std_dev_till_end

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
                # print("Smoothed data frame:")
                # print(df_smoothed.head()) # always starts with bunch of NaNs at start of averaging window
                print("Getting start of SSS based on smoothed signal:")
            


            # Get start of SSS based on where the value of the flux in the smoothed signal
            # is within tol_fac of the mean of the remaining signal, 
            # where tol_fac = factor * the rolling std dev of the stationary signal

            # At each location, compute the mean of the remaining signal
            stdv_frac = 0.1
            # tol_fac = stdv_frac * std_dev_stat_signal
            n_pts_smoothed = len(df_smoothed)
            mean_vals = np.empty((n_pts_smoothed,),dtype=float)
            # stdv_vals = np.empty((n_pts_smoothed,),dtype=float)
            for i in range(n_pts_smoothed):
                mean_vals[i] = np.mean(df_smoothed[col].iloc[i:])
                # stdv_vals[i] = np.std(df_smoothed[col].iloc[i:])
            # Check where the current value of the smoothed signal is within tol_fac of the mean of the remaining signal
            deviation = np.abs(df_smoothed[col] - mean_vals)
            # tol_fac = stdv_frac * (df_smoothed[col+'_std'] + 1.e-6*abs(mean_vals[0])) # stdv_frac * rolling std dev + a fudge factor in case there is no noise
            tol_fac = stdv_frac * (df_smoothed[col+'_std_till_end'] + 0.1*abs(mean_vals[0])) # stdv_frac * std dev till end + a fudge factor in case there is no noise (and to guard against factor going to zero when std dev goes to 0 at end of signal)
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
                true_sss_start_index = max(0, int(crit_met_index - 0.8*rolling_window)) # adjust for rolling window
                sss_start_time = df_smoothed['time'].iloc[true_sss_start_index] # adjust for rolling window

                if self._verbosity > 0:
                    print(f"Index where criterion is met: {crit_met_index}")
                    print(f"Rolling window: {rolling_window}")
                    print(f"time where criterion is met: {criterion_time}")
                    print(f"time at start of SSS (criterion met - 0.8*rolling window): {sss_start_time}")
                

                # Plot deviation and tolerance vs. time
                if self._verbosity > 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_smoothed['time'], deviation, label='Deviation', color='blue')
                    plt.plot(df_smoothed['time'], tolerance, label='Tolerance', color='orange')
                    plt.axvline(x=criterion_time, color='g', linestyle='--', label=f"Small Change Criterion Met")
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
        data_stream: full Data Stream
        col: string with label of variable we are processing
        stats: dictionary with statistics returned by process_data_steam()
        label: label to use in title of graph
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
