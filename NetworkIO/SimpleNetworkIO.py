import pandas as pd
import random

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO

from typing import Optional


class AllTrafficNetworkIO(BaseNetworkIO):
    """
    Allows all data to pass
    """
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return True


class NoTrafficNetworkIO(BaseNetworkIO):
    """
    Allows no data to pass
    """
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return False


class OnlyPortProtoNetworkIO(BaseNetworkIO):
    """
    Only allows a specific protocol to pass
    """
    def __init__(self, allowed_proto: Layer4Protocol, baseline_data: Optional[pd.DataFrame] = None):
        """
        :param allowed_proto: the protocol that's allowed to be sent over
        :param baseline_data: the baseline data of the communication on each protocol
        """
        super().__init__(baseline_data)
        self.allowed_proto: Layer4Protocol = allowed_proto

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return proto == self.allowed_proto

    def __str__(self) -> str:
        return f"Only{str(self.allowed_proto)}NetworkIO"


class NotPortProtoNetworkIO(BaseNetworkIO):
    """
    Allows all data to pass except for a specific protocol
    """
    def __init__(self, banned_proto: Layer4Protocol, baseline_data: Optional[pd.DataFrame] = None):
        """
        :param banned_proto: the protocol that's not allowed to be sent over
        :param baseline_data: the baseline data of the communication on each protocol
        """
        super().__init__(baseline_data)
        self.banned_proto: Layer4Protocol = banned_proto

    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        enforced_data = baseline_data.copy()

        if str(self.banned_proto) in baseline_data.index:
            enforced_data.loc[str(self.banned_proto)] *= 0

        return enforced_data

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return proto != self.banned_proto

    def __str__(self) -> str:
        return f"Not{str(self.banned_proto)}NetworkIO"


class RandomXPercentFailNetworkIO(BaseNetworkIO):
    """
    Has a random chance to fail the passage of data
    """
    def __init__(self, fail_chance: float, baseline_data: Optional[pd.DataFrame] = None):
        """

        :param fail_chance: the chance to fail the passage of each `send`. must have 0.0 <= fail_chance <= 1.0
        :param baseline_data: the baseline data of the communication on each protocol
        """
        super().__init__(baseline_data)
        self.fail_chance: float = fail_chance

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return random.uniform(0, 1) > self.fail_chance

    def __str__(self) -> str:
        return f"Random{self.fail_chance * 100}PercentFailNetworkIO"
