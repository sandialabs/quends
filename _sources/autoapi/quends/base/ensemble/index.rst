quends.base.ensemble
====================

.. py:module:: quends.base.ensemble


Classes
-------

.. autoapisummary::

   quends.base.ensemble.Ensemble


Module Contents
---------------

.. py:class:: Ensemble(data_streams: List[quends.base.data_stream.DataStream])

   Manages an ensemble of DataStream instances, enabling multi-stream analysis.

   Provides methods for:
     - Simple accessors (.head, .get_member, .members).
     - Identifying common variables across streams.
     - Generating an average-ensemble stream aligned to the shortest time grid.
     - Applying DataStream methods (mean, uncertainty, CI, ESS) at the ensemble level
       via three techniques: average-ensemble, aggregate-then-statistics, and weighted.
     - Tracking per-stream and ensemble metadata histories for reproducibility.


   .. py:attribute:: data_streams


   .. py:method:: head(n=5)

      Retrieve the first `n` rows from each DataStream member.

      :Parameters: **n** (*int*) -- Number of rows to return per stream.

      :returns: *Dict[int, pandas.DataFrame]* -- Mapping from member index to its DataFrame head.



   .. py:method:: get_member(index)

      Fetch a specific ensemble member by index.

      :Parameters: **index** (*int*) -- Zero-based index of the DataStream in the ensemble.

      :returns: *DataStream*

      :raises IndexError: If `index` is out of bounds.



   .. py:method:: members()

      List all ensemble members.

      :returns: *List[DataStream]*



   .. py:method:: common_variables()

      Identify variable columns shared by all members, excluding 'time'.

      :returns: *List[str]*



   .. py:method:: summary()

      Print and return a structured summary of ensemble members.

      Includes each member's sample count, column list, and head rows.

      :returns: *dict* --

                { 'n_members': int,
                  'common_variables': List[str],
                  'members': { 'Member i': { 'n_samples': int,
                                              'columns': List[str],
                                              'head': dict } } }



   .. py:method:: compute_average_ensemble(members: List[quends.base.data_stream.DataStream] = None)

      Build a DataStream whose columns are the elementwise mean across members,
      aligned on the shortest time grid.

      :Parameters: **members** (*List[DataStream], optional*) -- Subset of streams to average; defaults to all.

      :returns: *DataStream*

      :raises ValueError: If no streams are provided.



   .. py:method:: resample_to_short_intervals(short_df: pandas.DataFrame, long_df: pandas.DataFrame)

      Align `long_df` onto `short_df.time` by block-averaging between boundaries.

      :Parameters: * **short_df** (*pandas.DataFrame*) -- Reference DataFrame with the shortest time series.
                   * **long_df** (*pandas.DataFrame*) -- Stream to resample.

      :returns: *pandas.DataFrame* -- Resampled data matching `short_df.time`.



   .. py:method:: collect_histories(ds_list: List[quends.base.data_stream.DataStream])
      :staticmethod:


      Gather `_history` lists from each DataStream in `ds_list`.

      :Parameters: **ds_list** (*List[DataStream]*) -- Streams whose histories to collect.

      :returns: *List[List[dict]]*



   .. py:method:: trim(column_name: str, window_size: int = 10, start_time: float = 0.0, method: str = 'std', threshold: float = None, robust: bool = True) -> Dict

      Apply steady-state trimming to each member on `column_name`.

      :returns: *dict* --

                { 'results': Ensemble or None,
                  'metadata': Dict[str, Any] }



   .. py:method:: is_stationary(columns) -> Dict

      Test stationarity for `columns` across all members.

      :returns: *dict* --

                { 'results': {Member i: {col: bool or error}},
                  'metadata': {Member i: history} }



   .. py:method:: effective_sample_size(column_names=None, alpha: float = 0.05, technique: int = 0) -> Dict

      Compute classic ESS via three techniques:
        0 - on average-ensemble
        1 - on concatenated aggregate
        2 - per-member then aggregate

      :returns: *dict* -- { 'results': ..., 'metadata': ... }



   .. py:method:: ess_robust(column_names=None, rank_normalize=True, min_samples=8, return_relative=False, technique=0)

      Compute robust ESS (rank-based) via three techniques.

      :returns: *dict* -- { 'results': ..., 'metadata': ... }



   .. py:method:: mean(column_name=None, method='non-overlapping', window_size=None, technique=0)

      Compute ensemble mean via three techniques:
        0 - average-ensemble
        1 - aggregate-then-statistics
        2 - weighted per-member

      :returns: *dict* -- { 'results': ..., 'metadata': ... }



   .. py:method:: mean_uncertainty(column_name=None, ddof=1, method='non-overlapping', window_size=None, technique=0)

      Compute SEM via three techniques (0: average, 1: aggregate, 2: weighted).

      :returns: *dict*



   .. py:method:: confidence_interval(column_name=None, ddof=1, method='non-overlapping', window_size=None, technique=0)

      Compute 95% CI via three techniques.

      :returns: *dict*



   .. py:method:: compute_statistics(column_name=None, ddof=1, method='non-overlapping', window_size=None, technique=0)

      Aggregate mean, SEM, CI, and ±1std across the ensemble.

      :returns: *dict* -- { 'results': {col: {stats}}, 'metadata': {...} }



