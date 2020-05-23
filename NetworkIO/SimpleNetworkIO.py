import pandas as pd
import random

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO

from typing import Optional


class AllTrafficNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return True


class NoTrafficNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return False


class OnlyPortProtoNetworkIO(BaseNetworkIO):
    def __init__(self, allowed_proto: Layer4Protocol, baseline_data: Optional[pd.DataFrame] = None):
        super().__init__(baseline_data)
        self.allowed_proto: Layer4Protocol = allowed_proto

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return proto == self.allowed_proto

    def __str__(self) -> str:
        return f"Only{str(self.allowed_proto)}NetworkIO"


class RandomXPercentFailNetworkIO(BaseNetworkIO):
    def __init__(self, fail_chance: float, baseline_data: Optional[pd.DataFrame] = None):
        super().__init__(baseline_data)
        self.fail_chance: float = fail_chance

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return random.uniform(0, 1) > self.fail_chance

    def __str__(self) -> str:
        return f"Random{self.fail_chance * 100}PercentFailNetworkIO"
