from abc import ABC, abstractmethod


class TrimStrategy(ABC):
    """
    Default Trim Strategy
    """

    @abstractmethod
    def detect_steady_state(self, data, column_name, window_size, robust):
        """
        Abstract base class for steady-state detection strategies
        """
        pass


class StdTrimStrategy(TrimStrategy):
    """
    StdTrimStrategy
    """

    def detect_steady_state(self):
        """
        Docstring for detect_steady_state
        """
        pass


class ThresholdTrimStrategy(TrimStrategy):
    """
    ThreshodlTrimStrategy
    """

    def detect_steady_state(self):
        """
        Docstring for detect_steady_state
        """
        pass


class RollingVarianceTrimStrategy(TrimStrategy):
    """
    RollingVarianceTrimStrategy
    """

    def detect_steady_state(self):
        """
        Docstring for detect_steady_state
        """
        pass


class Trim:
    """
    Docstring for Trim
    """

    def __init__(self, strategy: TrimStrategy) -> None:
        """
        Initialize with specific trim strategy

        :param self: Description
        :param strategy: Description
        :type strategy: TrimStrategy
        """

        self._strategy = strategy

    @property
    def strategy(self) -> TrimStrategy:
        """
        Get the current strategy

        :param self: Description
        :return: Description
        :rtype: TrimStrategy
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: TrimStrategy) -> None:
        """
        Set a new strategy

        :param self: Description
        :param strategy: Description
        :type strategy: TrimStrategy
        """
        self._strategy = strategy

    def trim(self):
        """
        Trim the data by removing transient portion before steady state
        (Make this an option??)
        and then use default trim
        """


# what would go in datastream
class DataStream:
    def trim(
        self,
        column_name,
        batch_size,
        start_time,
        method="std",
        threshold=None,
        robust=None,
    ):
        if method == "std":
            strategy = StdTrimStrategy()
        elif method == "threshold":
            strategy = ThresholdTrimStrategy()
        elif method == "rolling_var":
            strategy = RollingVarianceTrimStrategy()
        else:
            raise ValueError("Unknown trim method")

        trimmer = Trim(strategy)
        self.data = trimmer.trim(
            data=self.data,
            column_name=column_name,
            window_size=batch_size,
            start_time=start_time,
            threshold=threshold,
            robust=robust,
        )
