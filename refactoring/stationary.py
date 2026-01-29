from .data_stream import DataStream
from .operations import DataStreamOperation


class MakeStationaryOperation(DataStreamOperation):
    """from make_stationarity"""

    def __init__(self, column, n_pts_orig, workflow):
        self.column = column
        self.n_pts_orig = n_pts_orig
        self.workflow = workflow

    def _apply(self, data_stream: DataStream) -> DataStream:
        """Iteratively drops data until stationary"""

        return
