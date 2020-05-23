import pandas as pd

import abc

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum


class BaseNetworkIO(abc.ABC):
    def __init__(self, baseline_data: pd.DataFrame):
        self.baseline_data: pd.DataFrame = baseline_data

    @abc.abstractmethod
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        pass

    def __call__(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return self.send(data, proto, data_texture)

    @abc.abstractmethod
    def __str__(self) -> str:
        return type(self).__name__
