import pandas as pd

import abc

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum

from typing import Optional


class BaseNetworkIO(abc.ABC):
    def __init__(self, baseline_data: Optional[pd.DataFrame] = None):
        self.baseline_data: pd.DataFrame = baseline_data

    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        return baseline_data.copy()

    @abc.abstractmethod
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        pass

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data

    def __call__(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return self.send(data, proto, data_texture)

    def __str__(self) -> str:
        return type(self).__name__

    def reset(self):
        pass
