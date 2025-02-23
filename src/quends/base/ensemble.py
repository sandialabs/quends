import pandas as pd
import numpy as np
import json
import math
from scipy.optimize import curve_fit
from statsmodels.robust.scale import mad
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, acf
from quends.base.data_stream import DataStream


class Ensemble:
    """
    Represents an ensemble of DataStream objects.
    
    Attributes:
        data_streams (list): List of DataStream instances.
    """
    def __init__(self, data_streams):
        # Validate input type and content.
        if not isinstance(data_streams, list):
            raise ValueError("data_streams must be a list of DataStream objects.")
        if not data_streams:
            raise ValueError("The list of data streams is empty.")
        if not all(isinstance(ds, DataStream) for ds in data_streams):
            raise ValueError("Each ensemble member must be a DataStream instance.")
        self.data_streams = data_streams

    def __len__(self):
        """Return the number of ensemble members."""
        return len(self.data_streams)

    def head(self, n=5):
        """
        Return the first n rows of each ensemble member.
        
        Returns:
            dict: Keys are member indices; values are DataFrame heads.
        """
        return {i: ds.head(n) for i, ds in enumerate(self.data_streams)}

    def get_member(self, index):
        """Return a specific ensemble member by index."""
        return self.data_streams[index]

    def members(self):
        """Return the list of all ensemble members."""
        return self.data_streams

    def common_variables(self):
        """
        Return a sorted list of variable names common to all members (excluding 'time').
        """
        # Create a list of sets (each member's columns except 'time')
        all_cols = [set(ds.df.columns) - {"time"} for ds in self.data_streams]
        # Return the sorted intersection if available
        return sorted(list(set.intersection(*all_cols))) if all_cols else []

    def summary(self):
        """
        Print and return a summary of the ensemble: number of members, common variables, 
        and a brief summary (number of samples, columns, head) for each member.
        
        Returns:
            dict: Overall summary dictionary.
        """
        summary_dict = {
            f"Member {i}": {
                "n_samples": len(ds.df),
                "columns": list(ds.df.columns),
                "head": ds.head().to_dict(orient="list")
            }
            for i, ds in enumerate(self.data_streams)
        }
        overall_summary = {
            "n_members": len(self.data_streams),
            "common_variables": self.common_variables(),
            "members": summary_dict
        }
        print("Ensemble Summary:")
        print(f"Number of ensemble members: {len(self.data_streams)}")
        print("Common variables:", self.common_variables())
        for member, info in summary_dict.items():
            print(f"\n{member}:")
            print(f"  Number of samples: {info['n_samples']}")
            print(f"  Columns: {info['columns']}")
            print("  Head:")
            print(pd.DataFrame(info['head']))
        return overall_summary

    def trim(self, column_name, window_size=10, start_time=0.0, method="std", threshold=None, robust=True):
        """
        Trim each ensemble member by calling its trim() method.
        Only members that return a non-None trimmed DataStream are kept.
        
        Args:
            column_name (str or list): Column(s) to trim.
            window_size (int): Window size for analysis.
            start_time (float): Start time threshold.
            method (str): Method to detect steady state ("std", "threshold", or "rolling_variance").
            threshold (float): Threshold for detection (if required).
            robust (bool): Use robust statistics if True.
        
        Returns:
            Ensemble: A new Ensemble with trimmed members (or None if none succeed).
        """
        trimmed = [ds.trim(column_name, window_size=window_size, start_time=start_time,
                             method=method, threshold=threshold, robust=robust)
                   for ds in self.data_streams]
        trimmed_members = [t for t in trimmed if t is not None]
        if not trimmed_members:
            print("None of the ensemble members could be trimmed with the specified parameters.")
            return None
        return Ensemble(trimmed_members)

    def is_stationary(self, columns):
        """
        Check stationarity of the specified columns in each ensemble member.
        
        Args:
            columns (str or list): Column(s) to test.
        
        Returns:
            dict: Keys are member indices, values are test results.
        """
        return {f"Member {i}": ds.is_stationary(columns) for i, ds in enumerate(self.data_streams)}

    def effective_sample_size(self, column_names=None, alpha=0.05):
        """
        Compute the effective sample size (ESS) for specified columns in each member.
        
        Args:
            column_names (str or list): Columns to analyze. Defaults to all except 'time'.
            max_plot_lags (int): Maximum lags for ACF.
            alpha (float): Significance level.
        
        Returns:
            dict: ESS results per member.
        """
        return {f"Member {i}": ds.effective_sample_size(column_names, alpha)
                for i, ds in enumerate(self.data_streams)}

    # ---------------- Ensemble Statistical Methods ----------------
    # Technique 0: Process each member individually.
    # Technique 1: Aggregate raw data across members and return both aggregated & individual.
    # Technique 2: Aggregate individual member statistics using weighted formulas.

    def mean(self, column_name=None, method="non-overlapping", window_size=None, technique=0):
        """
        Compute the mean for the ensemble.
        
        Technique 0: Returns individual member means.
        Technique 1: Aggregates raw data from members, then computes mean.
        Technique 2: Computes a weighted mean using each member's processed data length as weight.
        
        Returns:
            dict: Dictionary with "Individual Members" and "Member Ensemble" keys.
        """
        if technique == 0:
            return {f"Member {i}": ds.mean(column_name, method=method, window_size=window_size)
                    for i, ds in enumerate(self.data_streams)}
        elif technique == 1:
            individual = {f"Member {i}": ds.mean(column_name, method=method, window_size=window_size)
                          for i, ds in enumerate(self.data_streams)}
            cols = ([column_name] if isinstance(column_name, str)
                    else self.common_variables() if column_name is None else column_name)
            aggregated = {col: pd.concat([ds.df[col] for ds in self.data_streams
                                           if col in ds.df.columns and not ds.df[col].empty],
                                          axis=0, ignore_index=True)
                          for col in cols}
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                agg_result = ds_agg.mean(list(agg_df.columns), method=method, window_size=window_size)
            else:
                agg_result = {}
            return {"Member Ensemble": agg_result, "Individual Members": individual}
        elif technique == 2:
            # Technique 2: Weighted average of individual member means.
            member_means = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                res = ds.mean(column_name, method=method, window_size=window_size)
                member_means[key] = res
                member_weights[key] = {}
                # Determine columns for weighting.
                if column_name is None:
                    cols = ds.df.columns.drop("time")
                elif isinstance(column_name, str):
                    cols = [column_name]
                else:
                    cols = column_name
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if col_data.empty:
                            member_weights[key][col] = 0
                        else:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
            if column_name is None:
                agg_cols = self.common_variables()
            elif isinstance(column_name, str):
                agg_cols = [column_name]
            else:
                agg_cols = column_name
            ensemble_mean = {}
            for col in agg_cols:
                values, weights = [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if key in member_means and col in member_means[key]:
                        values.append(member_means[key][col]["mean"])
                        weights.append(member_weights.get(key, {}).get(col, 0))
                if values and np.sum(weights) > 0:
                    ensemble_mean[col] = np.sum(np.array(weights) * np.array(values)) / np.sum(weights)
            return {"Individual Members": member_means, "Member Ensemble": ensemble_mean}

    def mean_uncertainty(self, column_name=None, ddof=1, method="non-overlapping", window_size=None, technique=0):
        """
        Compute the mean uncertainty for the ensemble.
        
        Technique 0: Returns individual member uncertainties.
        Technique 1: Aggregates raw data and computes uncertainty on the concatenated Series.
        Technique 2: Computes a weighted aggregation using:
            - For each member, the weight is the number of processed data points.
            - The ensemble uncertainty is computed as:
                SEM = sqrt(Σ nᵢ (uᵢ² + (μᵢ - μ_w)²) / Σ nᵢ) / sqrt(Σ nᵢ)
              and the weighted average uncertainty is also returned.
        
        Returns:
            dict: {"Individual Members": ..., "Member Ensemble": {col: {"mean_uncertainty": SEM,
                                                                           "mean_uncertainty_average": weighted_avg}}}
        """
        if technique == 0:
            return {f"Member {i}": ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                    for i, ds in enumerate(self.data_streams)}
        elif technique == 1:
            individual = {f"Member {i}": ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                          for i, ds in enumerate(self.data_streams)}
            cols = ([column_name] if isinstance(column_name, str)
                    else self.common_variables() if column_name is None else column_name)
            aggregated = {col: pd.concat([ds.df[col] for ds in self.data_streams
                                           if col in ds.df.columns and not ds.df[col].empty],
                                          axis=0, ignore_index=True)
                          for col in cols}
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                agg_result = ds_agg.mean_uncertainty(list(agg_df.columns), ddof=ddof, method=method, window_size=window_size)
            else:
                agg_result = {}
            # Compute arithmetic average of individual uncertainties.
            member_unc_vals = {}
            for col in cols:
                vals = [individual[f"Member {i}"][col]["mean uncertainty"]
                        for i in range(len(self.data_streams))
                        if col in individual[f"Member {i}"]]
                if vals:
                    member_unc_vals[col] = np.mean(vals)
            return {"Member Ensemble": {"mean_uncertainty": agg_result,
                                          "mean_uncertainty_average": member_unc_vals},
                    "Individual Members": individual}
        elif technique == 2:
            # Technique 2: Weighted aggregation for mean uncertainty.
            member_unc = {}
            member_means = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                res_mean = ds.mean(column_name, method=method, window_size=window_size)
                res_unc = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                member_means[key] = res_mean
                member_unc[key] = res_unc
                member_weights[key] = {}
                if column_name is None:
                    cols = ds.df.columns.drop("time")
                elif isinstance(column_name, str):
                    cols = [column_name]
                else:
                    cols = column_name
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if col_data.empty:
                            member_weights[key][col] = 0
                        else:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
            if column_name is None:
                agg_cols = self.common_variables()
            elif isinstance(column_name, str):
                agg_cols = [column_name]
            else:
                agg_cols = column_name
            ensemble_unc = {}
            for col in agg_cols:
                # For weighted aggregation, we need individual member means and uncertainties.
                mu_vals, u_vals, weights = [], [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if key in member_means and col in member_means[key] and key in member_unc and col in member_unc[key]:
                        mu_vals.append(member_means[key][col]["mean"])
                        u_vals.append(member_unc[key][col]["mean uncertainty"])
                        weights.append(member_weights.get(key, {}).get(col, 0))
                if mu_vals and np.sum(weights) > 0:
                    weights = np.array(weights)
                    mu_vals = np.array(mu_vals)
                    u_vals = np.array(u_vals)
                    # Weighted average of member means.
                    weighted_mu = np.sum(weights * mu_vals) / np.sum(weights)
                    # Weighted average of member uncertainties.
                    weighted_avg_unc = np.sum(weights * u_vals) / np.sum(weights)
                    # Weighted variance: sum_i [w_i * (u_i^2 + (mu_i - weighted_mu)^2)] / sum_i(w_i)
                    weighted_var = np.sum(weights * (u_vals**2 + (mu_vals - weighted_mu)**2)) / np.sum(weights)
                    # Ensemble uncertainty defined as the SEM of the individual uncertainties:
                    ensemble_sem = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_unc[col] = {"mean_uncertainty": ensemble_sem,
                                         "mean_uncertainty_average": weighted_avg_unc}
            return {"Individual Members": member_unc, "Member Ensemble": ensemble_unc}

    def confidence_interval(self, column_name=None, ddof=1, method="non-overlapping", window_size=None, technique=0):
        """
        Compute confidence intervals for the ensemble.
        
        Technique 0: Returns individual members' CIs.
        Technique 1: Aggregates raw data then computes CI.
        Technique 2: Computes weighted ensemble mean and uncertainty and then derives CI as:
                   CI = (weighted_mean - 1.96*weighted_uncertainty, weighted_mean + 1.96*weighted_uncertainty)
        Returns:
            dict: {"Individual Members": ..., "Member Ensemble": ...}
        """
        if technique == 0:
            results = {}
            for i, ds in enumerate(self.data_streams):
                m_res = ds.mean(column_name, method=method, window_size=window_size)
                u_res = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                ci = {col: (m_res[col]["mean"] - 1.96 * u_res[col]["mean uncertainty"],
                            m_res[col]["mean"] + 1.96 * u_res[col]["mean uncertainty"])
                      for col in ds._get_columns(column_name)
                      if col in m_res and col in u_res}
                results[f"Member {i}"] = ci
            return results
        elif technique == 1:
            individual = {}
            for i, ds in enumerate(self.data_streams):
                m_res = ds.mean(column_name, method=method, window_size=window_size)
                u_res = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                ci = {col: (m_res[col]["mean"] - 1.96 * u_res[col]["mean uncertainty"],
                            m_res[col]["mean"] + 1.96 * u_res[col]["mean uncertainty"])
                      for col in ds._get_columns(column_name)
                      if col in m_res and col in u_res}
                individual[f"Member {i}"] = ci
            cols = ([column_name] if isinstance(column_name, str)
                    else self.common_variables() if column_name is None else column_name)
            aggregated = {col: pd.concat([ds.df[col] for ds in self.data_streams
                                           if col in ds.df.columns and not ds.df[col].empty],
                                     axis=0, ignore_index=True)
                          for col in cols}
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                m_agg = ds_agg.mean(list(agg_df.columns), method=method, window_size=window_size)
                u_agg = ds_agg.mean_uncertainty(list(agg_df.columns), ddof=ddof, method=method, window_size=window_size)
                ci_agg = {col: (m_agg[col]["mean"] - 1.96 * u_agg[col]["mean uncertainty"],
                                m_agg[col]["mean"] + 1.96 * u_agg[col]["mean uncertainty"])
                          for col in agg_df.columns
                          if col in m_agg and col in u_agg}
            else:
                ci_agg = {}
            return {"Member Ensemble": ci_agg, "Individual Members": individual}
        elif technique == 2:
            # Technique 2: Weighted aggregation of individual members' CIs.
            individual = {}
            member_cis = {}
            for i, ds in enumerate(self.data_streams):
                m_res = ds.mean(column_name, method=method, window_size=window_size)
                u_res = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                ci = {col: (m_res[col]["mean"] - 1.96 * u_res[col]["mean uncertainty"],
                            m_res[col]["mean"] + 1.96 * u_res[col]["mean uncertainty"])
                      for col in ds._get_columns(column_name)
                      if col in m_res and col in u_res}
                individual[f"Member {i}"] = ci
                member_cis[f"Member {i}"] = ci
            cols = ([column_name] if isinstance(column_name, str)
                    else self.common_variables() if column_name is None else column_name)
            # Use weighted mean and weighted uncertainty from technique 2.
            ensemble_mean = self.mean(column_name, method=method, window_size=window_size, technique=2)["Member Ensemble"]
            ensemble_unc = self.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size, technique=2)["Member Ensemble"]
            ensemble_ci = {}
            for col in cols:
                if col in ensemble_mean and col in ensemble_unc:
                    m = ensemble_mean[col]
                    u = ensemble_unc[col]["mean_uncertainty"]
                    ensemble_ci[col] = (m - 1.96 * u, m + 1.96 * u)
            return {"Individual Members": member_cis, "Member Ensemble": ensemble_ci}

    def compute_statistics(self, column_name=None, ddof=1, method="non-overlapping", window_size=None, technique=0):
        """
        Compute statistics (mean, mean uncertainty, confidence interval, and ±1 std) for each member,
        then aggregate them for the ensemble.
        
        Technique 0: Process each member individually.
        Technique 1: Aggregate raw data then compute statistics.
        Technique 2: Compute individual member statistics and aggregate using weighted formulas.
        
        Returns:
            dict: {"Individual Members": {...}, "Member Ensemble": {...}}
        """
        if technique == 0:
            results = {}
            for i, ds in enumerate(self.data_streams):
                m_res = ds.mean(column_name, method=method, window_size=window_size)
                u_res = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                ci_res = ds.confidence_interval(column_name, ddof=ddof, method=method, window_size=window_size)
                stats = {col: {
                            "mean": m_res[col]["mean"],
                            "mean_uncertainty": u_res[col]["mean uncertainty"],
                            "confidence_interval": ci_res[col]["confidence interval"],
                            "pm_std": (m_res[col]["mean"] - u_res[col]["mean uncertainty"],
                                       m_res[col]["mean"] + u_res[col]["mean uncertainty"])
                          }
                         for col in ds._get_columns(column_name)
                         if col in m_res and col in u_res and col in ci_res}
                results[f"Member {i}"] = stats
            return results
        elif technique == 1:
            individual = {}
            for i, ds in enumerate(self.data_streams):
                m_res = ds.mean(column_name, method=method, window_size=window_size)
                u_res = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                ci_res = ds.confidence_interval(column_name, ddof=ddof, method=method, window_size=window_size)
                stats = {col: {
                            "mean": m_res[col]["mean"],
                            "mean_uncertainty": u_res[col]["mean uncertainty"],
                            "confidence_interval": ci_res[col]["confidence interval"],
                            "pm_std": (m_res[col]["mean"] - u_res[col]["mean uncertainty"],
                                       m_res[col]["mean"] + u_res[col]["mean uncertainty"])
                          }
                         for col in ds._get_columns(column_name)
                         if col in m_res and col in u_res and col in ci_res}
                individual[f"Member {i}"] = stats
            cols = ([column_name] if isinstance(column_name, str)
                    else self.common_variables() if column_name is None else column_name)
            aggregated = {col: pd.concat([ds.df[col] for ds in self.data_streams 
                                           if col in ds.df.columns and not ds.df[col].empty],
                                          axis=0, ignore_index=True)
                          for col in cols}
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                m_agg = ds_agg.mean(list(agg_df.columns), method=method, window_size=window_size)
                u_agg = ds_agg.mean_uncertainty(list(agg_df.columns), ddof=ddof, method=method, window_size=window_size)
                ci_agg = ds_agg.confidence_interval(list(agg_df.columns), ddof=ddof, method=method, window_size=window_size)
                agg_result = {col: {
                                "mean": m_agg[col]["mean"],
                                "mean_uncertainty": u_agg[col]["mean uncertainty"],
                                "confidence_interval": ci_agg[col]["confidence interval"],
                                "pm_std": (m_agg[col]["mean"] - u_agg[col]["mean uncertainty"],
                                           m_agg[col]["mean"] + u_agg[col]["mean uncertainty"])
                              }
                              for col in agg_df.columns
                              if col in m_agg and col in u_agg and col in ci_agg}
            else:
                agg_result = {}
            return {"Member Ensemble": agg_result, "Individual Members": individual}
        elif technique == 2:
            # Technique 2: Compute individual member statistics and then aggregate them using weighted formulas.
            member_stats = {}
            for i, ds in enumerate(self.data_streams):
                m_res = ds.mean(column_name, method=method, window_size=window_size)
                u_res = ds.mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)
                ci_res = ds.confidence_interval(column_name, ddof=ddof, method=method, window_size=window_size)
                stats = {col: {
                            "mean": m_res[col]["mean"],
                            "mean_uncertainty": u_res[col]["mean uncertainty"],
                            "confidence_interval": ci_res[col]["confidence interval"],
                            "pm_std": (m_res[col]["mean"] - u_res[col]["mean uncertainty"],
                                       m_res[col]["mean"] + u_res[col]["mean uncertainty"])
                          }
                         for col in ds._get_columns(column_name)
                         if col in m_res and col in u_res and col in ci_res}
                member_stats[f"Member {i}"] = stats

            cols = ([column_name] if isinstance(column_name, str)
                    else self.common_variables() if column_name is None else column_name)
            # For each column, aggregate the statistics using weighted formulas.
            ensemble_stats = {}
            for col in cols:
                means = []
                uncs = []
                lowers = []
                uppers = []
                weights = []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if key in member_stats and col in member_stats[key]:
                        m_i = member_stats[key][col]["mean"]
                        u_i = member_stats[key][col]["mean_uncertainty"]
                        # Weight based on processed data length.
                        col_data = self.data_streams[i].df[col].dropna() if col in self.data_streams[i].df.columns else pd.Series()
                        w_i = 0
                        if not col_data.empty:
                            est_win = self.data_streams[i]._estimate_window(col, col_data, window_size)
                            processed = self.data_streams[i]._process_column(col_data, est_win, method)
                            w_i = len(processed)
                        if w_i > 0:
                            means.append(m_i)
                            uncs.append(u_i)
                            weights.append(w_i)
                            lowers.append(member_stats[key][col]["confidence_interval"][0])
                            uppers.append(member_stats[key][col]["confidence_interval"][1])
                if means and np.sum(weights) > 0:
                    weights = np.array(weights)
                    means = np.array(means)
                    uncs = np.array(uncs)
                    weighted_mean = np.sum(weights * means) / np.sum(weights)
                    # Weighted variance: sum_i[w_i * (u_i^2 + (m_i - weighted_mean)^2)] / sum_i[w_i]
                    weighted_var = np.sum(weights * (uncs**2 + (means - weighted_mean)**2)) / np.sum(weights)
                    ensemble_unc = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_stats[col] = {
                        "mean": weighted_mean,
                        "mean_uncertainty": ensemble_unc,
                        "mean_uncertainty_average": np.sum(weights * uncs) / np.sum(weights),
                        "confidence_interval": (np.mean(lowers), np.mean(uppers)),
                        "pm_std": (weighted_mean - ensemble_unc, weighted_mean + ensemble_unc)
                    }
            return {"Individual Members": member_stats, "Member Ensemble": ensemble_stats}