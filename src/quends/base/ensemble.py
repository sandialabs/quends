from typing import Dict, List

import numpy as np
import pandas as pd

from quends.base.data_stream import DataStream
<<<<<<< HEAD
...


from quends.base.data_stream import DataStream
...


from quends.base.data_stream import DataStream

"""
Module: ensemble.py

Defines the Ensemble class for handling collections of DataStream objects,
providing ensemble-level aggregation, statistical summaries, and metadata history.
"""


"""
Module: ensemble.py

Defines the Ensemble class for handling collections of DataStream objects,
providing ensemble-level aggregation, statistical summaries, and metadata history.
"""


class Ensemble:
    """
    Manages an ensemble of DataStream instances, enabling multi-stream analysis.

    Provides methods for:
      - Simple accessors (.head, .get_member, .members).
      - Identifying common variables across streams.
      - Generating an average-ensemble stream aligned to the shortest time grid.
      - Applying DataStream methods (mean, uncertainty, CI, ESS) at the ensemble level
        via three techniques: average-ensemble, aggregate-then-statistics, and weighted.
      - Tracking per-stream and ensemble metadata histories for reproducibility.
    """

    def __init__(self, data_streams: List[DataStream]):
        """
        Initialize the ensemble with a non-empty list of DataStream objects.

        Parameters
        ----------
        data_streams : List[DataStream]
            List of DataStream instances to include.

        Raises
        ------
        ValueError
            If input is not a list, is empty, or contains non-DataStream items.
        """
=======

"""
Module: ensemble.py

Defines the Ensemble class for handling collections of DataStream objects,
providing ensemble-level aggregation, statistical summaries, and metadata history.
"""
class Ensemble:
    """
    Manages an ensemble of DataStream instances, enabling multi-stream analysis.

    Provides methods for:
      - Simple accessors (.head, .get_member, .members).
      - Identifying common variables across streams.
      - Generating an average-ensemble stream aligned to the shortest time grid.
      - Applying DataStream methods (mean, uncertainty, CI, ESS) at the ensemble level
        via three techniques: average-ensemble, aggregate-then-statistics, and weighted.
      - Tracking per-stream and ensemble metadata histories for reproducibility.
    """

    def __init__(self, data_streams: List[DataStream]):
<<<<<<< HEAD
>>>>>>> 04a7df1 (Update: improvements to ensemble.py)
=======
        """
        Initialize the ensemble with a non-empty list of DataStream objects.

        Parameters
        ----------
        data_streams : List[DataStream]
            List of DataStream instances to include.

        Raises
        ------
        ValueError
            If input is not a list, is empty, or contains non-DataStream items.
        """
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
        if not isinstance(data_streams, list) or not data_streams:
            raise ValueError("Provide a non-empty list of DataStream objects.")
        if not all(isinstance(ds, DataStream) for ds in data_streams):
            raise ValueError("All ensemble members must be DataStream instances.")
        self.data_streams = data_streams

    def __len__(self):
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
        """
        Number of members in the ensemble.

        Returns
        -------
        int
        """
<<<<<<< HEAD
        return len(self.data_streams)

    def head(self, n=5):
        """
        Retrieve the first `n` rows from each DataStream member.

        Parameters
        ----------
        n : int
            Number of rows to return per stream.

        Returns
        -------
        Dict[int, pandas.DataFrame]
            Mapping from member index to its DataFrame head.
        """
        return {i: ds.head(n) for i, ds in enumerate(self.data_streams)}

    def get_member(self, index):
        """
        Fetch a specific ensemble member by index.

        Parameters
        ----------
        index : int
            Zero-based index of the DataStream in the ensemble.

        Returns
        -------
        DataStream

        Raises
        ------
        IndexError
            If `index` is out of bounds.
        """
        return self.data_streams[index]

    def members(self):
        """
        List all ensemble members.

        Returns
        -------
        List[DataStream]
        """
        return self.data_streams

    def common_variables(self):
        """
        Identify variable columns shared by all members, excluding 'time'.

        Returns
        -------
        List[str]
        """
        all_cols = [set(ds.df.columns) - {"time"} for ds in self.data_streams]
        if not all_cols:
            return []
        return sorted(list(set.intersection(*all_cols)))

    def summary(self):
        """
        Print and return a structured summary of ensemble members.

        Includes each member's sample count, column list, and head rows.

        Returns
        -------
        dict
            { 'n_members': int,
              'common_variables': List[str],
              'members': { 'Member i': { 'n_samples': int,
                                          'columns': List[str],
                                          'head': dict } } }
        """
=======
=======
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
        return len(self.data_streams)

    def head(self, n=5):
        """
        Retrieve the first `n` rows from each DataStream member.

        Parameters
        ----------
        n : int
            Number of rows to return per stream.

        Returns
        -------
        Dict[int, pandas.DataFrame]
            Mapping from member index to its DataFrame head.
        """
        return {i: ds.head(n) for i, ds in enumerate(self.data_streams)}

    def get_member(self, index):
        """
        Fetch a specific ensemble member by index.

        Parameters
        ----------
        index : int
            Zero-based index of the DataStream in the ensemble.

        Returns
        -------
        DataStream

        Raises
        ------
        IndexError
            If `index` is out of bounds.
        """
        return self.data_streams[index]

    def members(self):
        """
        List all ensemble members.

        Returns
        -------
        List[DataStream]
        """
        return self.data_streams

    def common_variables(self):
        """
        Identify variable columns shared by all members, excluding 'time'.

        Returns
        -------
        List[str]
        """
        all_cols = [set(ds.df.columns) - {"time"} for ds in self.data_streams]
        if not all_cols:
            return []
        return sorted(list(set.intersection(*all_cols)))

    def summary(self):
<<<<<<< HEAD
>>>>>>> 04a7df1 (Update: improvements to ensemble.py)
=======
        """
        Print and return a structured summary of ensemble members.

        Includes each member's sample count, column list, and head rows.

        Returns
        -------
        dict
            { 'n_members': int,
              'common_variables': List[str],
              'members': { 'Member i': { 'n_samples': int,
                                          'columns': List[str],
                                          'head': dict } } }
        """
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
        summary_dict = {
            f"Member {i}": {
                "n_samples": len(ds.df),
                "columns": list(ds.df.columns),
                "head": ds.head().to_dict(orient="list"),
            }
            for i, ds in enumerate(self.data_streams)
        }
        overall_summary = {
            "n_members": len(self.data_streams),
            "common_variables": self.common_variables(),
            "members": summary_dict,
        }
        print("Ensemble Summary:")
        print(f"Number of ensemble members: {len(self.data_streams)}")
        print("Common variables:", self.common_variables())
        for member, info in summary_dict.items():
            print(f"\n{member}:")
            print(f"  Number of samples: {info['n_samples']}")
            print(f"  Columns: {info['columns']}")
            print("  Head:")
            print(pd.DataFrame(info["head"]))
        return overall_summary

<<<<<<< HEAD
<<<<<<< HEAD
    # ========== Core DataStream Method Shortcuts ==========
    def _mean(
        self,
        ds: DataStream,
        column_name=None,
        method="non-overlapping",
        window_size=None,
    ):
<<<<<<< HEAD
        return ds._mean(column_name, method=method, window_size=window_size)

    def _mean_uncertainty(
        self,
        ds: DataStream,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        return ds._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )

    def _confidence_interval(
        self,
        ds: DataStream,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        return ds._confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )

    def _classic_ess(self, ds: DataStream, column_names=None, alpha=0.05):
        return ds.effective_sample_size(column_names, alpha)

    def _robust_ess(self, ds: DataStream, column_names=None, **kwargs):
        return ds.ess_robust(column_names, **kwargs)

    # ========== Average-Ensemble Construction ==========
    def compute_average_ensemble(self, members: List[DataStream] = None):
        """
        Build a DataStream whose columns are the elementwise mean across members,
        aligned on the shortest time grid.

        Parameters
        ----------
        members : List[DataStream], optional
            Subset of streams to average; defaults to all.

        Returns
        -------
        DataStream

        Raises
        ------
        ValueError
            If no streams are provided.
        """
        data_streams = members if members is not None else self.data_streams
        data_frames: Dict[str, pd.DataFrame] = {
            f"Member {i}": ds.df for i, ds in enumerate(data_streams)
        }
        if not data_frames:
            raise ValueError("No data streams provided for ensemble averaging.")
        shortest_df = min(data_frames.values(), key=lambda df: len(df))
        short_times = shortest_df["time"].values
        resampled = {
            name: self.resample_to_short_intervals(shortest_df, df)
            for name, df in data_frames.items()
        }
        ensemble_avg = shortest_df[["time"]].copy().reset_index(drop=True)
        for col in shortest_df.columns:
            if col == "time":
                continue
            arrays = [df[col].to_numpy() for df in resampled.values()]
            ensemble_avg[col] = np.mean(arrays, axis=0)
        return DataStream(ensemble_avg)

    def resample_to_short_intervals(
        self, short_df: pd.DataFrame, long_df: pd.DataFrame
    ):
        """
        Align `long_df` onto `short_df.time` by block-averaging between boundaries.

        Parameters
        ----------
        short_df : pandas.DataFrame
            Reference DataFrame with the shortest time series.
        long_df : pandas.DataFrame
            Stream to resample.

        Returns
        -------
        pandas.DataFrame
            Resampled data matching `short_df.time`.
        """
        short_times = short_df["time"].values
        long_times = long_df["time"].values
        idx = np.searchsorted(long_times, short_times)
        out = pd.DataFrame({"time": short_times.copy()})
        for col in long_df.columns:
            if col == "time":
                continue
            vals = long_df[col].values
            means = [
                np.nanmean(vals[start:end]) if end > start else np.nan
                for start, end in zip(idx[:-1], idx[1:])
            ]
            tail = vals[idx[-1] :]
            means.append(np.nanmean(tail) if tail.size else np.nan)
            out[col] = means
        return out

    # ========== Utility ==========
    @staticmethod
    def collect_histories(ds_list: List[DataStream]):
        """
        Gather `_history` lists from each DataStream in `ds_list`.

        Parameters
        ----------
        ds_list : List[DataStream]
            Streams whose histories to collect.

        Returns
        -------
        List[List[dict]]
        """
        return [getattr(ds, "_history", []) for ds in ds_list]

    # ========== TRIM ==========
    def trim(self, column_name, batch_size=10, start_time=0.0, method="std", threshold=None, robust=True):
        trimmed = [
            ds.trim(
                column_name,
                batch_size=batch_size,
                start_time=start_time,
                method=method,
                threshold=threshold,
                robust=robust,
            )
            for ds in self.data_streams
        ]
        # Only keep non-empty, valid DataStreams
        trimmed_members = [t for t in trimmed if t is not None and (hasattr(t, 'df') and not t.df.empty)]
        if not trimmed_members:
            raise ValueError("No ensemble members survived trimming (all failed or empty)!")
        return Ensemble(trimmed_members)




    # ========== IS_STATIONARY ==========
    def is_stationary(self, columns) -> Dict:
        """
        Test stationarity for `columns` across all members.

        Returns
        -------
        dict
            { 'results': {Member i: {col: bool or error}},
              'metadata': {Member i: history} }
        """
        results, meta = {}, {}
        for i, ds in enumerate(self.data_streams):
            r = ds.is_stationary(columns)
            results[f"Member {i}"] = r
            meta[f"Member {i}"] = getattr(ds, "_history", None)
        return {"results": results, "metadata": meta}

    # ========== EFFECTIVE SAMPLE SIZE ==========
    def effective_sample_size(
        self, column_names=None, alpha: float = 0.05, technique: int = 0
    ) -> Dict:
        """
        Compute classic ESS via three techniques:
          0 - on average-ensemble
          1 - on concatenated aggregate
          2 - per-member then aggregate

        Returns
        -------
        dict
            { 'results': ..., 'metadata': ... }
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._classic_ess(avg_ds, column_names, alpha)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names]
                if isinstance(column_names, str)
                else self.common_variables() if column_names is None else column_names
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._classic_ess(ds_agg, list(agg_df.columns), alpha)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Per-member ESS
            per_member_results = {}
            per_member_meta = {}
            for i, ds in enumerate(self.data_streams):
                res = self._classic_ess(ds, column_names, alpha)
                per_member_results[f"Member {i}"] = (
                    res.get("results") if isinstance(res, dict) else res
                )
                per_member_meta[f"Member {i}"] = (
                    res.get("metadata", None) if isinstance(res, dict) else None
                )
            # Aggregate: mean, harmonic mean, etc.
            ess_vals = []
            for v in per_member_results.values():
                if isinstance(v, dict):
                    for ess in v.values():
                        if isinstance(ess, (int, float)) and not np.isnan(ess):
                            ess_vals.append(ess)
            agg_ess = np.nanmean(ess_vals) if ess_vals else np.nan
            result = {"ensemble_ess": agg_ess, "individual_ess": per_member_results}
            metadata["per_member"] = per_member_meta
        return {"results": result, "metadata": metadata}

    # ========== ESS ROBUST ==========
    def ess_robust(
        self,
        column_names=None,
        rank_normalize=True,
        min_samples=8,
        return_relative=False,
        technique=0,
    ):
        """
        Compute robust ESS (rank-based) via three techniques.

        Returns
        -------
        dict
            { 'results': ..., 'metadata': ... }
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._robust_ess(
                avg_ds,
                column_names,
                rank_normalize=rank_normalize,
                min_samples=min_samples,
                return_relative=return_relative,
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names]
                if isinstance(column_names, str)
                else self.common_variables() if column_names is None else column_names
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._robust_ess(
                    ds_agg,
                    list(agg_df.columns),
                    rank_normalize=rank_normalize,
                    min_samples=min_samples,
                    return_relative=return_relative,
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Per-member robust ESS
            per_member_results = {}
            per_member_meta = {}
            for i, ds in enumerate(self.data_streams):
                res = self._robust_ess(
                    ds,
                    column_names,
                    rank_normalize=rank_normalize,
                    min_samples=min_samples,
                    return_relative=return_relative,
                )
                per_member_results[f"Member {i}"] = (
                    res.get("results") if isinstance(res, dict) else res
                )
                per_member_meta[f"Member {i}"] = (
                    res.get("metadata", None) if isinstance(res, dict) else None
                )
            # Aggregate: mean, harmonic mean, etc.
            ess_vals = []
            for v in per_member_results.values():
                if isinstance(v, dict):
                    for ess in v.values():
                        if isinstance(ess, (int, float)) and not np.isnan(ess):
                            ess_vals.append(ess)
            agg_ess = np.nanmean(ess_vals) if ess_vals else np.nan
            result = {
                "ensemble_robust_ess": agg_ess,
                "individual_robust_ess": per_member_results,
            }
            metadata["per_member"] = per_member_meta
        return {"results": result, "metadata": metadata}

    # ========== MEAN ==========
    def mean(
        self, column_name=None, method="non-overlapping", window_size=None, technique=0
    ):
        """
        Compute ensemble mean via three techniques:
          0 - average-ensemble
          1 - aggregate-then-statistics
          2 - weighted per-member

        Returns
        -------
        dict
            { 'results': ..., 'metadata': ... }
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._mean(avg_ds, column_name, method, window_size)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._mean(ds_agg, list(agg_df.columns), method, window_size)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Per-member means and weights
            member_means = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                member_means[key] = self._mean(ds, column_name, method, window_size)
                member_weights[key] = {}
                cols = (
                    ds.df.columns.drop("time")
                    if column_name is None
                    else [column_name] if isinstance(column_name, str) else column_name
                )
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if not col_data.empty:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
                        else:
                            member_weights[key][col] = 0
            agg_cols = (
                self.common_variables()
                if column_name is None
                else ([column_name] if isinstance(column_name, str) else column_name)
            )
            ensemble_mean = {}
            for col in agg_cols:
                values, weights = [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if key in member_means and col in member_means[key]:
                        values.append(member_means[key][col]["mean"])
                        weights.append(member_weights.get(key, {}).get(col, 0))
                if values and np.sum(weights) > 0:
                    ensemble_mean[col] = np.sum(
                        np.array(weights) * np.array(values)
                    ) / np.sum(weights)
            metadata["individual"] = {
                f"Member {i}": getattr(ds, "_history", None)
                for i, ds in enumerate(self.data_streams)
            }
            result = {
                "Member Ensemble": ensemble_mean,
                "Individual Members": member_means,
            }
        return {"results": result, "metadata": metadata}

    # ========== MEAN UNCERTAINTY ==========
    def mean_uncertainty(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        technique=0,
    ):
        """
        Compute SEM via three techniques (0: average, 1: aggregate, 2: weighted).

        Returns
        -------
        dict
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._mean_uncertainty(
                avg_ds, column_name, ddof, method, window_size
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._mean_uncertainty(
                    ds_agg, list(agg_df.columns), ddof, method, window_size
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            member_unc = {}
            member_means = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                member_means[key] = self._mean(ds, column_name, method, window_size)
                member_unc[key] = self._mean_uncertainty(
                    ds, column_name, ddof, method, window_size
                )
                member_weights[key] = {}
                cols = (
                    ds.df.columns.drop("time")
                    if column_name is None
                    else [column_name] if isinstance(column_name, str) else column_name
                )
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if not col_data.empty:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
                        else:
                            member_weights[key][col] = 0
            agg_cols = (
                self.common_variables()
                if column_name is None
                else ([column_name] if isinstance(column_name, str) else column_name)
            )
            ensemble_unc = {}
            for col in agg_cols:
                mu_vals, u_vals, weights = [], [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if (
                        key in member_means
                        and col in member_means[key]
                        and key in member_unc
                        and col in member_unc[key]
                    ):
                        mu_vals.append(member_means[key][col]["mean"])
                        u_vals.append(member_unc[key][col]["mean_uncertainty"])
                        weights.append(member_weights.get(key, {}).get(col, 0))
                if mu_vals and np.sum(weights) > 0:
                    weights = np.array(weights)
                    mu_vals = np.array(mu_vals)
                    u_vals = np.array(u_vals)
                    weighted_mu = np.sum(weights * mu_vals) / np.sum(weights)
                    weighted_avg_unc = np.sum(weights * u_vals) / np.sum(weights)
                    weighted_var = np.sum(
                        weights * (u_vals**2 + (mu_vals - weighted_mu) ** 2)
                    ) / np.sum(weights)
                    ensemble_sem = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_unc[col] = {
                        "mean_uncertainty": ensemble_sem,
                        "mean_uncertainty_average": weighted_avg_unc,
                    }
            metadata["individual"] = {
                f"Member {i}": getattr(ds, "_history", None)
                for i, ds in enumerate(self.data_streams)
            }
            result = {"Member Ensemble": ensemble_unc, "Individual Members": member_unc}
        return {"results": result, "metadata": metadata}

    # ========== CONFIDENCE INTERVAL ==========
    def confidence_interval(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        technique=0,
    ):
        """
        Compute 95% CI via three techniques.

        Returns
        -------
        dict
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._confidence_interval(
                avg_ds, column_name, ddof, method, window_size
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._confidence_interval(
                    ds_agg, list(agg_df.columns), ddof, method, window_size
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            member_cis = {}
            member_means = {}
            member_unc = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                member_means[key] = self._mean(ds, column_name, method, window_size)
                member_unc[key] = self._mean_uncertainty(
                    ds, column_name, ddof, method, window_size
                )
                member_cis[key] = self._confidence_interval(
                    ds, column_name, ddof, method, window_size
                )
                member_weights[key] = {}
                cols = (
                    ds.df.columns.drop("time")
                    if column_name is None
                    else [column_name] if isinstance(column_name, str) else column_name
                )
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if not col_data.empty:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
                        else:
                            member_weights[key][col] = 0
            agg_cols = (
                self.common_variables()
                if column_name is None
                else ([column_name] if isinstance(column_name, str) else column_name)
            )
            ensemble_ci = {}
            for col in agg_cols:
                means, uncs, lowers, uppers, weights = [], [], [], [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if (
                        key in member_means
                        and col in member_means[key]
                        and key in member_unc
                        and col in member_unc[key]
                        and key in member_cis
                        and col in member_cis[key]
                    ):
                        m_i = member_means[key][col]["mean"]
                        u_i = member_unc[key][col]["mean_uncertainty"]
                        ci = member_cis[key][col].get(
                            "confidence_interval", (np.nan, np.nan)
                        )
                        w_i = member_weights[key][col]
                        means.append(m_i)
                        uncs.append(u_i)
                        weights.append(w_i)
                        lowers.append(ci[0])
                        uppers.append(ci[1])
                if means and np.sum(weights) > 0:
                    weights = np.array(weights)
                    means = np.array(means)
                    uncs = np.array(uncs)
                    weighted_mean = np.sum(weights * means) / np.sum(weights)
                    weighted_var = np.sum(
                        weights * (uncs**2 + (means - weighted_mean) ** 2)
                    ) / np.sum(weights)
                    ensemble_unc = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_ci[col] = (
                        weighted_mean - 1.96 * ensemble_unc,
                        weighted_mean + 1.96 * ensemble_unc,
                    )
            metadata["individual"] = {
                f"Member {i}": getattr(ds, "_history", None)
                for i, ds in enumerate(self.data_streams)
            }
            result = {"Member Ensemble": ensemble_ci, "Individual Members": member_cis}
        return {"results": result, "metadata": metadata}

    # ========== FULL STATISTICS ==========
    def compute_statistics(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        technique=0,
    ):
        """
        Aggregate mean, SEM, CI, and ±1std across the ensemble.

        Returns
        -------
        dict
            { 'results': {col: {stats}}, 'metadata': {...} }
        """
        mean_result = self.mean(column_name, method, window_size, technique)
        unc_result = self.mean_uncertainty(
            column_name, ddof, method, window_size, technique
        )
        ci_result = self.confidence_interval(
            column_name, ddof, method, window_size, technique
        )
        stats = {}
        # Key structure depends on technique; adapt below as needed
        if technique == 2:
            for key in mean_result["results"]["Member Ensemble"]:
                stats[key] = {
                    "mean": mean_result["results"]["Member Ensemble"][key],
                    "mean_uncertainty": unc_result["results"]["Member Ensemble"][key][
                        "mean_uncertainty"
                    ],
                    "mean_uncertainty_average": unc_result["results"][
                        "Member Ensemble"
                    ][key]["mean_uncertainty_average"],
                    "confidence_interval": ci_result["results"]["Member Ensemble"][key],
                    "pm_std": (
                        mean_result["results"]["Member Ensemble"][key]
                        - unc_result["results"]["Member Ensemble"][key][
                            "mean_uncertainty"
                        ],
                        mean_result["results"]["Member Ensemble"][key]
                        + unc_result["results"]["Member Ensemble"][key][
                            "mean_uncertainty"
                        ],
                    ),
                }
        else:
            keys = mean_result["results"].keys()
            for key in keys:
                stats[key] = {
                    "mean": (
                        mean_result["results"][key]["mean"]
                        if "mean" in mean_result["results"][key]
                        else mean_result["results"][key]
                    ),
                    "mean_uncertainty": (
                        unc_result["results"][key]["mean_uncertainty"]
                        if "mean_uncertainty" in unc_result["results"][key]
                        else unc_result["results"][key]
                    ),
                    "confidence_interval": (
                        ci_result["results"][key]["confidence_interval"]
                        if "confidence_interval" in ci_result["results"][key]
                        else ci_result["results"][key]
                    ),
                    "pm_std": (
                        (
                            mean_result["results"][key]["mean"]
                            - unc_result["results"][key]["mean_uncertainty"]
                            if "mean" in mean_result["results"][key]
                            and "mean_uncertainty" in unc_result["results"][key]
                            else np.nan
                        ),
                        (
                            mean_result["results"][key]["mean"]
                            + unc_result["results"][key]["mean_uncertainty"]
                            if "mean" in mean_result["results"][key]
                            and "mean_uncertainty" in unc_result["results"][key]
                            else np.nan
                        ),
                    ),
                }
        metadata = {
            "mean": mean_result["metadata"],
            "mean_uncertainty": unc_result["metadata"],
            "confidence_interval": ci_result["metadata"],
        }
        return {"results": stats, "metadata": metadata}

<<<<<<< HEAD

# End of class
=======
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
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
                        col_data = (
                            self.data_streams[i].df[col].dropna()
                            if col in self.data_streams[i].df.columns
                            else pd.Series()
                        )
                        w_i = 0
                        if not col_data.empty:
                            est_win = self.data_streams[i]._estimate_window(
                                col, col_data, window_size
                            )
                            processed = self.data_streams[i]._process_column(
                                col_data, est_win, method
                            )
                            w_i = len(processed)
                        if w_i > 0:
                            means.append(m_i)
                            uncs.append(u_i)
                            weights.append(w_i)
                            lowers.append(
                                member_stats[key][col]["confidence_interval"][0]
                            )
                            uppers.append(
                                member_stats[key][col]["confidence_interval"][1]
                            )
                if means and np.sum(weights) > 0:
                    weights = np.array(weights)
                    means = np.array(means)
                    uncs = np.array(uncs)
                    weighted_mean = np.sum(weights * means) / np.sum(weights)
                    # Weighted variance: sum_i[w_i * (u_i^2 + (m_i - weighted_mean)^2)] / sum_i[w_i]
                    weighted_var = np.sum(
                        weights * (uncs**2 + (means - weighted_mean) ** 2)
                    ) / np.sum(weights)
                    ensemble_unc = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_stats[col] = {
                        "mean": weighted_mean,
                        "mean_uncertainty": ensemble_unc,
                        "mean_uncertainty_average": np.sum(weights * uncs)
                        / np.sum(weights),
                        "confidence_interval": (np.mean(lowers), np.mean(uppers)),
                        "pm_std": (
                            weighted_mean - ensemble_unc,
                            weighted_mean + ensemble_unc,
                        ),
                    }
            return {
                "Individual Members": member_stats,
                "Member Ensemble": ensemble_stats,
            }

    def resample_to_short_intervals(self,
                                    short_df: pd.DataFrame,
                                    long_df: pd.DataFrame):
=======
    # ---------------- Core Internal Statistical Methods ----------------
=======
    # ========== Core DataStream Method Shortcuts ==========
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
    def _mean(self, ds: DataStream, column_name=None, method="non-overlapping", window_size=None):
=======
>>>>>>> 2d15506 (update documentation with autoapi)
        return ds._mean(column_name, method=method, window_size=window_size)

    def _mean_uncertainty(
        self,
        ds: DataStream,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        return ds._mean_uncertainty(
            column_name, ddof=ddof, method=method, window_size=window_size
        )

    def _confidence_interval(
        self,
        ds: DataStream,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
    ):
        return ds._confidence_interval(
            column_name, ddof=ddof, method=method, window_size=window_size
        )

    def _classic_ess(self, ds: DataStream, column_names=None, alpha=0.05):
        return ds.effective_sample_size(column_names, alpha)

    def _robust_ess(self, ds: DataStream, column_names=None, **kwargs):
        return ds.ess_robust(column_names, **kwargs)

    # ========== Average-Ensemble Construction ==========
    def compute_average_ensemble(self, members: List[DataStream] = None):
        """
        Build a DataStream whose columns are the elementwise mean across members,
        aligned on the shortest time grid.

        Parameters
        ----------
        members : List[DataStream], optional
            Subset of streams to average; defaults to all.

        Returns
        -------
        DataStream

        Raises
        ------
        ValueError
            If no streams are provided.
        """
        data_streams = members if members is not None else self.data_streams
        data_frames: Dict[str, pd.DataFrame] = {
            f"Member {i}": ds.df for i, ds in enumerate(data_streams)
        }
        if not data_frames:
            raise ValueError("No data streams provided for ensemble averaging.")
        shortest_df = min(data_frames.values(), key=lambda df: len(df))
        short_times = shortest_df["time"].values
        resampled = {
            name: self.resample_to_short_intervals(shortest_df, df)
            for name, df in data_frames.items()
        }
        ensemble_avg = shortest_df[["time"]].copy().reset_index(drop=True)
        for col in shortest_df.columns:
            if col == "time":
                continue
            arrays = [df[col].to_numpy() for df in resampled.values()]
            ensemble_avg[col] = np.mean(arrays, axis=0)
        return DataStream(ensemble_avg)

<<<<<<< HEAD
    def resample_to_short_intervals(self, short_df: pd.DataFrame, long_df: pd.DataFrame):
<<<<<<< HEAD
>>>>>>> 04a7df1 (Update: improvements to ensemble.py)
=======
=======
    def resample_to_short_intervals(
        self, short_df: pd.DataFrame, long_df: pd.DataFrame
    ):
>>>>>>> 2d15506 (update documentation with autoapi)
        """
        Align `long_df` onto `short_df.time` by block-averaging between boundaries.

        Parameters
        ----------
        short_df : pandas.DataFrame
            Reference DataFrame with the shortest time series.
        long_df : pandas.DataFrame
            Stream to resample.

        Returns
        -------
        pandas.DataFrame
            Resampled data matching `short_df.time`.
        """
<<<<<<< HEAD
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
        short_times = short_df['time'].values
        long_times = long_df['time'].values
=======
        short_times = short_df["time"].values
        long_times = long_df["time"].values
>>>>>>> 2d15506 (update documentation with autoapi)
        idx = np.searchsorted(long_times, short_times)
        out = pd.DataFrame({"time": short_times.copy()})
        for col in long_df.columns:
            if col == "time":
                continue
            vals = long_df[col].values
            means = [
                np.nanmean(vals[start:end]) if end > start else np.nan
                for start, end in zip(idx[:-1], idx[1:])
            ]
            tail = vals[idx[-1] :]
            means.append(np.nanmean(tail) if tail.size else np.nan)
            out[col] = means
        return out

    # ========== Utility ==========
    @staticmethod
    def collect_histories(ds_list: List[DataStream]):
        """
        Gather `_history` lists from each DataStream in `ds_list`.

        Parameters
        ----------
        ds_list : List[DataStream]
            Streams whose histories to collect.

        Returns
        -------
        List[List[dict]]
        """
        return [getattr(ds, "_history", []) for ds in ds_list]

    # ========== TRIM ==========
    def trim(
        self,
        column_name: str,
        window_size: int = 10,
        start_time: float = 0.0,
        method: str = "std",
        threshold: float = None,
        robust: bool = True,
    ) -> Dict:
        """
        Apply steady-state trimming to each member on `column_name`.

        Returns
        -------
        dict
            { 'results': Ensemble or None,
              'metadata': Dict[str, Any] }
        """
        trimmed = [
            ds.trim(
                column_name,
                batch_size=window_size,
                start_time=start_time,
                method=method,
                threshold=threshold,
                robust=robust,
            )
            for ds in self.data_streams
        ]
        trimmed_members = [t["results"] for t in trimmed if t["results"] is not None]
        trimmed_meta = {f"Member {i}": t.get("metadata") for i, t in enumerate(trimmed)}
        if not trimmed_members:
            return {"results": None, "metadata": trimmed_meta}
        return {"results": Ensemble(trimmed_members), "metadata": trimmed_meta}

    # ========== IS_STATIONARY ==========
    def is_stationary(self, columns) -> Dict:
        """
        Test stationarity for `columns` across all members.

        Returns
        -------
        dict
            { 'results': {Member i: {col: bool or error}},
              'metadata': {Member i: history} }
        """
        results, meta = {}, {}
        for i, ds in enumerate(self.data_streams):
            r = ds.is_stationary(columns)
            results[f"Member {i}"] = r
            meta[f"Member {i}"] = getattr(ds, "_history", None)
        return {"results": results, "metadata": meta}

    # ========== EFFECTIVE SAMPLE SIZE ==========
    def effective_sample_size(
        self, column_names=None, alpha: float = 0.05, technique: int = 0
    ) -> Dict:
        """
        Compute classic ESS via three techniques:
          0 - on average-ensemble
          1 - on concatenated aggregate
          2 - per-member then aggregate

        Returns
        -------
        dict
            { 'results': ..., 'metadata': ... }
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._classic_ess(avg_ds, column_names, alpha)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names]
                if isinstance(column_names, str)
                else self.common_variables() if column_names is None else column_names
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._classic_ess(ds_agg, list(agg_df.columns), alpha)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Per-member ESS
            per_member_results = {}
            per_member_meta = {}
            for i, ds in enumerate(self.data_streams):
                res = self._classic_ess(ds, column_names, alpha)
                per_member_results[f"Member {i}"] = (
                    res.get("results") if isinstance(res, dict) else res
                )
                per_member_meta[f"Member {i}"] = (
                    res.get("metadata", None) if isinstance(res, dict) else None
                )
            # Aggregate: mean, harmonic mean, etc.
            ess_vals = []
            for v in per_member_results.values():
                if isinstance(v, dict):
                    for ess in v.values():
                        if isinstance(ess, (int, float)) and not np.isnan(ess):
                            ess_vals.append(ess)
            agg_ess = np.nanmean(ess_vals) if ess_vals else np.nan
            result = {"ensemble_ess": agg_ess, "individual_ess": per_member_results}
            metadata["per_member"] = per_member_meta
        return {"results": result, "metadata": metadata}

    # ========== ESS ROBUST ==========
    def ess_robust(
        self,
        column_names=None,
        rank_normalize=True,
        min_samples=8,
        return_relative=False,
        technique=0,
    ):
        """
        Compute robust ESS (rank-based) via three techniques.

        Returns
        -------
        dict
            { 'results': ..., 'metadata': ... }
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._robust_ess(
                avg_ds,
                column_names,
                rank_normalize=rank_normalize,
                min_samples=min_samples,
                return_relative=return_relative,
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names]
                if isinstance(column_names, str)
                else self.common_variables() if column_names is None else column_names
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._robust_ess(
                    ds_agg,
                    list(agg_df.columns),
                    rank_normalize=rank_normalize,
                    min_samples=min_samples,
                    return_relative=return_relative,
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Per-member robust ESS
            per_member_results = {}
            per_member_meta = {}
            for i, ds in enumerate(self.data_streams):
                res = self._robust_ess(
                    ds,
                    column_names,
                    rank_normalize=rank_normalize,
                    min_samples=min_samples,
                    return_relative=return_relative,
                )
                per_member_results[f"Member {i}"] = (
                    res.get("results") if isinstance(res, dict) else res
                )
                per_member_meta[f"Member {i}"] = (
                    res.get("metadata", None) if isinstance(res, dict) else None
                )
            # Aggregate: mean, harmonic mean, etc.
            ess_vals = []
            for v in per_member_results.values():
                if isinstance(v, dict):
                    for ess in v.values():
                        if isinstance(ess, (int, float)) and not np.isnan(ess):
                            ess_vals.append(ess)
            agg_ess = np.nanmean(ess_vals) if ess_vals else np.nan
            result = {
                "ensemble_robust_ess": agg_ess,
                "individual_robust_ess": per_member_results,
            }
            metadata["per_member"] = per_member_meta
        return {"results": result, "metadata": metadata}

<<<<<<< HEAD
<<<<<<< HEAD
        # wrap and return as a DataStream so we can perform other UQ analysis on it
        return DataStream(ensemble_avg)
>>>>>>> 4e28c83 (Update ensemble.py)
=======
    # -------------- Add other aggregate/statistical methods here (same pattern) ------------------

# End of Ensemble class
>>>>>>> 04a7df1 (Update: improvements to ensemble.py)
=======
    # ========== MEAN ==========
    def mean(
        self, column_name=None, method="non-overlapping", window_size=None, technique=0
    ):
        """
        Compute ensemble mean via three techniques:
          0 - average-ensemble
          1 - aggregate-then-statistics
          2 - weighted per-member

        Returns
        -------
        dict
            { 'results': ..., 'metadata': ... }
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._mean(avg_ds, column_name, method, window_size)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._mean(ds_agg, list(agg_df.columns), method, window_size)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Per-member means and weights
            member_means = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                member_means[key] = self._mean(ds, column_name, method, window_size)
                member_weights[key] = {}
                cols = (
                    ds.df.columns.drop("time")
                    if column_name is None
                    else [column_name] if isinstance(column_name, str) else column_name
                )
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if not col_data.empty:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
                        else:
                            member_weights[key][col] = 0
            agg_cols = (
                self.common_variables()
                if column_name is None
                else ([column_name] if isinstance(column_name, str) else column_name)
            )
            ensemble_mean = {}
            for col in agg_cols:
                values, weights = [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if key in member_means and col in member_means[key]:
                        values.append(member_means[key][col]["mean"])
                        weights.append(member_weights.get(key, {}).get(col, 0))
                if values and np.sum(weights) > 0:
                    ensemble_mean[col] = np.sum(
                        np.array(weights) * np.array(values)
                    ) / np.sum(weights)
            metadata["individual"] = {
                f"Member {i}": getattr(ds, "_history", None)
                for i, ds in enumerate(self.data_streams)
            }
            result = {
                "Member Ensemble": ensemble_mean,
                "Individual Members": member_means,
            }
        return {"results": result, "metadata": metadata}

    # ========== MEAN UNCERTAINTY ==========
    def mean_uncertainty(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        technique=0,
    ):
        """
        Compute SEM via three techniques (0: average, 1: aggregate, 2: weighted).

        Returns
        -------
        dict
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._mean_uncertainty(
                avg_ds, column_name, ddof, method, window_size
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._mean_uncertainty(
                    ds_agg, list(agg_df.columns), ddof, method, window_size
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            member_unc = {}
            member_means = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                member_means[key] = self._mean(ds, column_name, method, window_size)
                member_unc[key] = self._mean_uncertainty(
                    ds, column_name, ddof, method, window_size
                )
                member_weights[key] = {}
                cols = (
                    ds.df.columns.drop("time")
                    if column_name is None
                    else [column_name] if isinstance(column_name, str) else column_name
                )
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if not col_data.empty:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
                        else:
                            member_weights[key][col] = 0
            agg_cols = (
                self.common_variables()
                if column_name is None
                else ([column_name] if isinstance(column_name, str) else column_name)
            )
            ensemble_unc = {}
            for col in agg_cols:
                mu_vals, u_vals, weights = [], [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if (
                        key in member_means
                        and col in member_means[key]
                        and key in member_unc
                        and col in member_unc[key]
                    ):
                        mu_vals.append(member_means[key][col]["mean"])
                        u_vals.append(member_unc[key][col]["mean_uncertainty"])
                        weights.append(member_weights.get(key, {}).get(col, 0))
                if mu_vals and np.sum(weights) > 0:
                    weights = np.array(weights)
                    mu_vals = np.array(mu_vals)
                    u_vals = np.array(u_vals)
                    weighted_mu = np.sum(weights * mu_vals) / np.sum(weights)
                    weighted_avg_unc = np.sum(weights * u_vals) / np.sum(weights)
                    weighted_var = np.sum(
                        weights * (u_vals**2 + (mu_vals - weighted_mu) ** 2)
                    ) / np.sum(weights)
                    ensemble_sem = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_unc[col] = {
                        "mean_uncertainty": ensemble_sem,
                        "mean_uncertainty_average": weighted_avg_unc,
                    }
            metadata["individual"] = {
                f"Member {i}": getattr(ds, "_history", None)
                for i, ds in enumerate(self.data_streams)
            }
            result = {"Member Ensemble": ensemble_unc, "Individual Members": member_unc}
        return {"results": result, "metadata": metadata}

    # ========== CONFIDENCE INTERVAL ==========
    def confidence_interval(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        technique=0,
    ):
        """
        Compute 95% CI via three techniques.

        Returns
        -------
        dict
        """
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._confidence_interval(
                avg_ds, column_name, ddof, method, window_size
            )
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name]
                if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat(
                    [
                        ds.df[col]
                        for ds in self.data_streams
                        if col in ds.df.columns and not ds.df[col].empty
                    ],
                    axis=0,
                    ignore_index=True,
                )
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._confidence_interval(
                    ds_agg, list(agg_df.columns), ddof, method, window_size
                )
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            member_cis = {}
            member_means = {}
            member_unc = {}
            member_weights = {}
            for i, ds in enumerate(self.data_streams):
                key = f"Member {i}"
                member_means[key] = self._mean(ds, column_name, method, window_size)
                member_unc[key] = self._mean_uncertainty(
                    ds, column_name, ddof, method, window_size
                )
                member_cis[key] = self._confidence_interval(
                    ds, column_name, ddof, method, window_size
                )
                member_weights[key] = {}
                cols = (
                    ds.df.columns.drop("time")
                    if column_name is None
                    else [column_name] if isinstance(column_name, str) else column_name
                )
                for col in cols:
                    if col in ds.df.columns:
                        col_data = ds.df[col].dropna()
                        if not col_data.empty:
                            est_win = ds._estimate_window(col, col_data, window_size)
                            processed = ds._process_column(col_data, est_win, method)
                            member_weights[key][col] = len(processed)
                        else:
                            member_weights[key][col] = 0
            agg_cols = (
                self.common_variables()
                if column_name is None
                else ([column_name] if isinstance(column_name, str) else column_name)
            )
            ensemble_ci = {}
            for col in agg_cols:
                means, uncs, lowers, uppers, weights = [], [], [], [], []
                for i in range(len(self.data_streams)):
                    key = f"Member {i}"
                    if (
                        key in member_means
                        and col in member_means[key]
                        and key in member_unc
                        and col in member_unc[key]
                        and key in member_cis
                        and col in member_cis[key]
                    ):
                        m_i = member_means[key][col]["mean"]
                        u_i = member_unc[key][col]["mean_uncertainty"]
                        ci = member_cis[key][col].get(
                            "confidence_interval", (np.nan, np.nan)
                        )
                        w_i = member_weights[key][col]
                        means.append(m_i)
                        uncs.append(u_i)
                        weights.append(w_i)
                        lowers.append(ci[0])
                        uppers.append(ci[1])
                if means and np.sum(weights) > 0:
                    weights = np.array(weights)
                    means = np.array(means)
                    uncs = np.array(uncs)
                    weighted_mean = np.sum(weights * means) / np.sum(weights)
                    weighted_var = np.sum(
                        weights * (uncs**2 + (means - weighted_mean) ** 2)
                    ) / np.sum(weights)
                    ensemble_unc = np.sqrt(weighted_var) / np.sqrt(np.sum(weights))
                    ensemble_ci[col] = (
                        weighted_mean - 1.96 * ensemble_unc,
                        weighted_mean + 1.96 * ensemble_unc,
                    )
            metadata["individual"] = {
                f"Member {i}": getattr(ds, "_history", None)
                for i, ds in enumerate(self.data_streams)
            }
            result = {"Member Ensemble": ensemble_ci, "Individual Members": member_cis}
        return {"results": result, "metadata": metadata}

    # ========== FULL STATISTICS ==========
    def compute_statistics(
        self,
        column_name=None,
        ddof=1,
        method="non-overlapping",
        window_size=None,
        technique=0,
    ):
        """
        Aggregate mean, SEM, CI, and ±1std across the ensemble.

        Returns
        -------
        dict
            { 'results': {col: {stats}}, 'metadata': {...} }
        """
        mean_result = self.mean(column_name, method, window_size, technique)
        unc_result = self.mean_uncertainty(
            column_name, ddof, method, window_size, technique
        )
        ci_result = self.confidence_interval(
            column_name, ddof, method, window_size, technique
        )
        stats = {}
        # Key structure depends on technique; adapt below as needed
        if technique == 2:
            for key in mean_result["results"]["Member Ensemble"]:
                stats[key] = {
                    "mean": mean_result["results"]["Member Ensemble"][key],
                    "mean_uncertainty": unc_result["results"]["Member Ensemble"][key][
                        "mean_uncertainty"
                    ],
                    "mean_uncertainty_average": unc_result["results"][
                        "Member Ensemble"
                    ][key]["mean_uncertainty_average"],
                    "confidence_interval": ci_result["results"]["Member Ensemble"][key],
                    "pm_std": (
                        mean_result["results"]["Member Ensemble"][key]
                        - unc_result["results"]["Member Ensemble"][key][
                            "mean_uncertainty"
                        ],
                        mean_result["results"]["Member Ensemble"][key]
                        + unc_result["results"]["Member Ensemble"][key][
                            "mean_uncertainty"
                        ],
                    ),
                }
        else:
            keys = mean_result["results"].keys()
            for key in keys:
                stats[key] = {
                    "mean": (
                        mean_result["results"][key]["mean"]
                        if "mean" in mean_result["results"][key]
                        else mean_result["results"][key]
                    ),
                    "mean_uncertainty": (
                        unc_result["results"][key]["mean_uncertainty"]
                        if "mean_uncertainty" in unc_result["results"][key]
                        else unc_result["results"][key]
                    ),
                    "confidence_interval": (
                        ci_result["results"][key]["confidence_interval"]
                        if "confidence_interval" in ci_result["results"][key]
                        else ci_result["results"][key]
                    ),
                    "pm_std": (
                        (
                            mean_result["results"][key]["mean"]
                            - unc_result["results"][key]["mean_uncertainty"]
                            if "mean" in mean_result["results"][key]
                            and "mean_uncertainty" in unc_result["results"][key]
                            else np.nan
                        ),
                        (
                            mean_result["results"][key]["mean"]
                            + unc_result["results"][key]["mean_uncertainty"]
                            if "mean" in mean_result["results"][key]
                            and "mean_uncertainty" in unc_result["results"][key]
                            else np.nan
                        ),
                    ),
                }
        metadata = {
            "mean": mean_result["metadata"],
            "mean_uncertainty": unc_result["metadata"],
            "confidence_interval": ci_result["metadata"],
        }
        return {"results": stats, "metadata": metadata}


# End of class
>>>>>>> 0565272 (Enhance DataStream statistics output and analysis options handling)
