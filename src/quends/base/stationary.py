from quends.base.history import DataStreamHistoryEntry

from .data_stream import DataStream
from .operations import DataStreamOperation


class MakeDataStreamStationaryOperation(DataStreamOperation):
    def __init__(
        self,
        column,
        n_pts_orig,
        *,
        operate_safe=None,
        n_pts_min=None,
        n_pts_frac_min=None,
        drop_fraction=None,
        verbosity=None,
    ):
        super().__init__(operation_name="make_stationary")
        self.column = column
        self.n_pts_orig = n_pts_orig
        self.is_stationary = None
        self.operate_safe = operate_safe
        self.n_pts_min = n_pts_min
        self.n_pts_frac_min = n_pts_frac_min
        self.drop_fraction = drop_fraction
        self.verbosity = verbosity

    def __call__(self, data_stream: DataStream, **kwargs) -> tuple[DataStream, bool]:
        """
        Override to handle tuple return value.
        Returns (DataStream, is_stationary) instead of just DataStream.
        """
        result_ds, is_stationary = self._apply(data_stream, **kwargs)

        if not is_stationary:
            empty_df = data_stream.data.iloc[0:0].copy()
            result_ds = DataStream(empty_df, history=data_stream.history)
            result_ds.message = f"Column '{self.column}' is not stationary"

        # Add history entry
        history_entry = DataStreamHistoryEntry(
            operation_name=self.name,
            parameters={
                "column": self.column,
                "n_pts_orig": self.n_pts_orig,
                "n_pts_final": len(result_ds.data),
                "stationary": is_stationary,
                **kwargs,
            },
        )
        result_ds._history.append(history_entry)

        return result_ds, is_stationary

    def _apply(self, data_stream: DataStream) -> DataStream:
        """
        Attempt to make the data stream into being stationary by removing an initial
        fraction of data.

        Parameters
        ----------
        col : str
        n_pts_orig : int
        workflow : RobustWorkflow

        Returns
        -------
        self : DataStream
        stationary : bool
        """
        col = self.column
        n_pts_orig = self.n_pts_orig

        ds = data_stream
        stationary = ds.is_stationary([col])[
            col
        ]  # is_stationary() returns dictionary. The value for key qoi tells us if it is stationary
        n_pts = len(ds.data)

        n_dropped = 0
        while (
            not stationary
            and not self.operate_safe
            and n_pts > self.n_pts_min
            and n_pts > self.n_pts_frac_min * n_pts_orig
        ):
            # See if we get a stationary stream if we drop some initial fraction of the data
            n_drop = int(n_pts * self.drop_fraction)
            df_shortened = ds.data.iloc[n_drop:]
            ds = DataStream(df_shortened)
            n_pts = len(ds.data)
            n_dropped = n_pts_orig - n_pts
            stationary = ds.is_stationary([col])[col]

            if self.verbosity > 0:
                if stationary:
                    print(
                        f"Data stream was not stationary, but is stationary after dropping first {n_dropped} points."
                    )
                else:
                    print(
                        f"Data stream is not stationary, even after dropping first {n_dropped} points."
                    )

        return ds, stationary
