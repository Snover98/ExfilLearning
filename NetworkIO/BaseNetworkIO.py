import pandas as pd

import abc

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum

from typing import Optional


class BaseNetworkIO(abc.ABC):
    """
    The base class for a network 'controller' that decides if a given data can be sent
    """
    def __init__(self, baseline_data: Optional[pd.DataFrame] = None):
        """
        :param baseline_data: the baseline data of the communication on each protocol
        """
        self.baseline_data: pd.DataFrame = baseline_data

    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces the NetworkIO's rules over the baseline data
        By default just copies it and should be overriden by more specific IOs
        :param baseline_data: the data to enforce the rules on
        :return: a new data with the NetworkIO's rules enforced on it
        """
        return baseline_data.copy()

    @abc.abstractmethod
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        """
        Decides whether or not it's legal to send the data.
        should be overridden by each subclass

        :param data: the data to send in bytes
        :param proto: the protocol over which the data will be sent
        :param data_texture: the data's texture
        :return: True if it's legal to send the data, False otherwise
        """
        pass

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data

    def __call__(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return self.send(data, proto, data_texture)

    def __str__(self) -> str:
        return type(self).__name__

    def reset(self):
        pass
