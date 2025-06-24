import numpy as np
import pandas as pd
from typing import List, Dict, Any

from quends.base.data_stream import DataStream

class Ensemble:
    """
    Represents an ensemble of DataStream objects with full reproducibility and
    flexible ensemble statistics: mean, mean uncertainty, confidence interval,
    classic and robust ESS, with full metadata propagation.
    """
    def __init__(self, data_streams: List[DataStream]):
        if not isinstance(data_streams, list) or not data_streams:
            raise ValueError("Provide a non-empty list of DataStream objects.")
        if not all(isinstance(ds, DataStream) for ds in data_streams):
            raise ValueError("All ensemble members must be DataStream instances.")
        self.data_streams = data_streams

    def __len__(self):
        return len(self.data_streams)

    def head(self, n=5):
        return {i: ds.head(n) for i, ds in enumerate(self.data_streams)}

    def get_member(self, index):
        return self.data_streams[index]

    def members(self):
        return self.data_streams

    def common_variables(self):
        all_cols = [set(ds.df.columns) - {"time"} for ds in self.data_streams]
        return sorted(list(set.intersection(*all_cols))) if all_cols else []

    def summary(self):
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

    # ---------------- Core Internal Statistical Methods ----------------
    def _mean(self, ds: DataStream, column_name=None, method="non-overlapping", window_size=None):
        return ds._mean(column_name, method=method, window_size=window_size)

    def _mean_uncertainty(self, ds: DataStream, column_name=None, ddof=1, method="non-overlapping", window_size=None):
        return ds._mean_uncertainty(column_name, ddof=ddof, method=method, window_size=window_size)

    def _confidence_interval(self, ds: DataStream, column_name=None, ddof=1, method="non-overlapping", window_size=None):
        return ds._confidence_interval(column_name, ddof=ddof, method=method, window_size=window_size)

    def _classic_ess(self, ds: DataStream, column_names=None, alpha=0.05):
        return ds.effective_sample_size(column_names, alpha)

    def _robust_ess(self, ds: DataStream, column_names=None, **kwargs):
        return ds.ess_robust(column_names, **kwargs)

    # --------- Utility: Average-ensemble DataStream (Technique 0) ---------
    def compute_average_ensemble(self, members: List[DataStream] = None):
        data_streams = members if members is not None else self.data_streams
        data_frames: Dict[str, pd.DataFrame] = {
            f"Member {i}": ds.df for i, ds in enumerate(data_streams)
        }
        if not data_frames:
            raise ValueError("No data streams provided for ensemble averaging.")
        shortest_df = min(data_frames.values(), key=lambda df: len(df))
        short_times = shortest_df['time'].values
        resampled = {
            name: self.resample_to_short_intervals(shortest_df, df)
            for name, df in data_frames.items()
        }
        ensemble_avg = shortest_df[['time']].copy().reset_index(drop=True)
        for col in shortest_df.columns:
            if col == 'time':
                continue
            arrays = [df[col].to_numpy() for df in resampled.values()]
            ensemble_avg[col] = np.mean(arrays, axis=0)
        return DataStream(ensemble_avg)

    def resample_to_short_intervals(self, short_df: pd.DataFrame, long_df: pd.DataFrame):
        short_times = short_df['time'].values
        long_times = long_df['time'].values
        indices = np.searchsorted(long_times, short_times)
        out = pd.DataFrame({'time': short_times.copy()})
        for col in long_df.columns:
            if col == 'time':
                continue
            vals = long_df[col].values
            means = []
            for start, end in zip(indices[:-1], indices[1:]):
                seg = vals[start:end]
                means.append(np.nanmean(seg) if seg.size else np.nan)
            tail = vals[indices[-1]:]
            means.append(np.nanmean(tail) if tail.size else np.nan)
            out[col] = means
        return out

    # --------- Metadata Utility ---------
    @staticmethod
    def collect_histories(ds_list: List[DataStream]):
        return [getattr(ds, "_history", []) for ds in ds_list]

    # --------- Ensemble Statistical Methods (with Metadata) ---------
    def mean(self, column_name=None, method="non-overlapping", window_size=None, technique=0):
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._mean(avg_ds, column_name, method, window_size)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            # Aggregate all data
            cols = (
                [column_name] if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat([ds.df[col] for ds in self.data_streams
                                if col in ds.df.columns and not ds.df[col].empty], axis=0, ignore_index=True)
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
            # Weighted mean of per-member
            member_means = []
            weights = []
            col_list = [column_name] if isinstance(column_name, str) else self.common_variables()
            for ds in self.data_streams:
                stat = self._mean(ds, column_name, method, window_size)
                for col in col_list:
                    if col in stat:
                        val = stat[col]["mean"]
                        n = len(ds.df)  # or use processed count
                        member_means.append(val)
                        weights.append(n)
            weights = np.array(weights)
            vals = np.array(member_means)
            if weights.sum() > 0:
                weighted = np.sum(weights * vals) / weights.sum()
            else:
                weighted = np.nan
            result = {"weighted_mean": weighted}
            metadata["per_member"] = self.collect_histories(self.data_streams)
        return {"results": result, "metadata": metadata}

    def mean_uncertainty(self, column_name=None, ddof=1, method="non-overlapping", window_size=None, technique=0):
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._mean_uncertainty(avg_ds, column_name, ddof, method, window_size)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name] if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat([ds.df[col] for ds in self.data_streams
                                if col in ds.df.columns and not ds.df[col].empty], axis=0, ignore_index=True)
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._mean_uncertainty(ds_agg, list(agg_df.columns), ddof, method, window_size)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            # Weighted variance for uncertainty
            member_unc = []
            weights = []
            col_list = [column_name] if isinstance(column_name, str) else self.common_variables()
            for ds in self.data_streams:
                stat = self._mean_uncertainty(ds, column_name, ddof, method, window_size)
                for col in col_list:
                    if col in stat:
                        val = stat[col]["mean_uncertainty"]
                        n = len(ds.df)  # or processed count
                        member_unc.append(val)
                        weights.append(n)
            weights = np.array(weights)
            vals = np.array(member_unc)
            if weights.sum() > 0:
                weighted = np.sqrt(np.sum(weights * vals**2) / weights.sum())
            else:
                weighted = np.nan
            result = {"weighted_mean_uncertainty": weighted}
            metadata["per_member"] = self.collect_histories(self.data_streams)
        return {"results": result, "metadata": metadata}

    def confidence_interval(self, column_name=None, ddof=1, method="non-overlapping", window_size=None, technique=0):
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._confidence_interval(avg_ds, column_name, ddof, method, window_size)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_name] if isinstance(column_name, str)
                else self.common_variables() if column_name is None else column_name
            )
            aggregated = {
                col: pd.concat([ds.df[col] for ds in self.data_streams
                                if col in ds.df.columns and not ds.df[col].empty], axis=0, ignore_index=True)
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._confidence_interval(ds_agg, list(agg_df.columns), ddof, method, window_size)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            mean_result = self.mean(column_name, method, window_size, technique=2)
            unc_result = self.mean_uncertainty(column_name, ddof, method, window_size, technique=2)
            if "results" in mean_result and "weighted_mean" in mean_result["results"] \
               and "results" in unc_result and "weighted_mean_uncertainty" in unc_result["results"]:
                mean = mean_result["results"]["weighted_mean"]
                unc = unc_result["results"]["weighted_mean_uncertainty"]
                ci = (mean - 1.96 * unc, mean + 1.96 * unc)
            else:
                ci = (np.nan, np.nan)
            result = {"ensemble_confidence_interval": ci}
            metadata["combined"] = {
                "mean_metadata": mean_result.get("metadata", {}),
                "unc_metadata": unc_result.get("metadata", {})
            }
        return {"results": result, "metadata": metadata}

    def effective_sample_size(self, column_names=None, alpha=0.05, technique=0):
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._classic_ess(avg_ds, column_names, alpha)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names] if isinstance(column_names, str)
                else self.common_variables() if column_names is None else column_names
            )
            aggregated = {
                col: pd.concat([ds.df[col] for ds in self.data_streams
                                if col in ds.df.columns and not ds.df[col].empty], axis=0, ignore_index=True)
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
            per_member_results = []
            for ds in self.data_streams:
                per_member_results.append(self._classic_ess(ds, column_names, alpha))
            # For the classic ESS, you may wish to aggregate (e.g., harmonic mean)
            ess_vals = []
            for r in per_member_results:
                # Grab first ESS value (may need to adapt for dict structure)
                v = list(r["results"].values())[0] if "results" in r and r["results"] else np.nan
                ess_vals.append(v)
            agg_ess = np.nanmean(ess_vals)
            result = {"ensemble_ess": agg_ess, "individual_ess": ess_vals}
            metadata["per_member"] = [r.get("metadata", {}) for r in per_member_results]
        return {"results": result, "metadata": metadata}

    def ess_robust(self, column_names=None, rank_normalize=True, min_samples=8, return_relative=False, technique=0):
        metadata = {}
        if technique == 0:
            avg_ds = self.compute_average_ensemble()
            result = self._robust_ess(avg_ds, column_names, rank_normalize=rank_normalize, min_samples=min_samples, return_relative=return_relative)
            metadata["average_ensemble"] = getattr(avg_ds, "_history", [])
        elif technique == 1:
            cols = (
                [column_names] if isinstance(column_names, str)
                else self.common_variables() if column_names is None else column_names
            )
            aggregated = {
                col: pd.concat([ds.df[col] for ds in self.data_streams
                                if col in ds.df.columns and not ds.df[col].empty], axis=0, ignore_index=True)
                for col in cols
            }
            if aggregated:
                agg_df = pd.concat(aggregated, axis=1)
                ds_agg = DataStream(agg_df)
                result = self._robust_ess(ds_agg, list(agg_df.columns), rank_normalize=rank_normalize, min_samples=min_samples, return_relative=return_relative)
                metadata["aggregated"] = getattr(ds_agg, "_history", [])
            else:
                result = {}
                metadata["aggregated"] = []
        elif technique == 2:
            per_member_results = []
            for ds in self.data_streams:
                per_member_results.append(
                    self._robust_ess(ds, column_names, rank_normalize=rank_normalize, min_samples=min_samples, return_relative=return_relative)
                )
            # For robust ESS, aggregate (e.g., harmonic mean or geometric mean)
            ess_vals = []
            for r in per_member_results:
                # Grab first ESS value (may need to adapt for dict structure)
                v = list(r["results"].values())[0] if "results" in r and r["results"] else np.nan
                ess_vals.append(v)
            agg_ess = np.nanmean(ess_vals)
            result = {"ensemble_robust_ess": agg_ess, "individual_robust_ess": ess_vals}
            metadata["per_member"] = [r.get("metadata", {}) for r in per_member_results]
        return {"results": result, "metadata": metadata}

    # -------------- Add other aggregate/statistical methods here (same pattern) ------------------

# End of Ensemble class
