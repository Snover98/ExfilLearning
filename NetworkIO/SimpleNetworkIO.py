import pandas as pd
import random

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO


class AllTrafficNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return True

    def __str__(self) -> str:
        return super().__str__()


class NoTrafficNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return False

    def __str__(self) -> str:
        return super().__str__()


class OnlyPortProtoNetworkIO(BaseNetworkIO):
    def __init__(self, baseline_data: pd.DataFrame, allowed_proto: Layer4Protocol):
        super().__init__(baseline_data)
        self.allowed_proto: Layer4Protocol = allowed_proto

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return proto == self.allowed_proto

    def __str__(self) -> str:
        return f"Only{str(self.allowed_proto)}NetworkIO"


class RandomXPercentFailNetworkIO(BaseNetworkIO):
    def __init__(self, baseline_data: pd.DataFrame, fail_chance: float):
        super().__init__(baseline_data)
        self.fail_chance: float = fail_chance

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return random.uniform(0, 1) > self.fail_chance

    def __str__(self) -> str:
        return f"Random{self.fail_chance * 100}PercentFailNetworkIO"
