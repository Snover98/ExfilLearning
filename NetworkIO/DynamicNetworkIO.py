import pandas as pd

from Protocols import Layer4Protocol, textual_protocols, str_to_layer4_proto
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO
from typing import Optional, Dict


class TextureNetworkIO(BaseNetworkIO):
    """
    Only allows data of the correct texture to be sent over each protocol.
    The valid textures for each protocol are described by `textual_protocols` from `Protocols`
    """
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        if proto in textual_protocols:
            proto_texture: DataTextureEnum = DataTextureEnum.textual
        else:
            proto_texture: DataTextureEnum = DataTextureEnum.binary

        return data_texture == proto_texture


class DataSizeWithinStdOfMeanForProtoNetworkIO(BaseNetworkIO):
    """
    For each protocol only allows data of whose size is
    within the `std_coef` * `proto_std` of the protocols average packet size
    """
    def __init__(self, baseline_data: Optional[pd.DataFrame] = None, std_coef: float = 1.0):
        """
        :param baseline_data: the baseline data of the communication on each protocol
        :param std_coef: the coefficient multiplying the std
        """
        super().__init__(baseline_data)
        self.std_coef: float = std_coef

    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        zeroes all protocols that have std == 0 and adjusts the minimum, maximum and median values to the enforced range

        :param baseline_data: the data to enforce the rules on
        :return: a new data with the NetworkIO's rules enforced on it
        """
        enforced_data = baseline_data.copy()

        enforced_data[baseline_data.packet_size_std_bytes == 0.0] *= 0

        std_size_range = enforced_data.packet_size_std_bytes * self.std_coef
        enforced_data.min_packet_size_bytes = enforced_data.avg_packet_size_bytes - std_size_range
        enforced_data.max_packet_size_bytes = enforced_data.avg_packet_size_bytes + std_size_range

        if not enforced_data.min_packet_size_bytes <= enforced_data.median_packet_size_bytes <= enforced_data.max_packet_size_bytes:
            enforced_data.median_packet_size_bytes = enforced_data.avg_packet_size_bytes

        for col in enforced_data.columns:
            enforced_data[col] = enforced_data[col].astype(baseline_data[col].dtype)

        return enforced_data

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        proto_baseline_data: pd.Series = self.baseline_data.loc[str(proto)]

        diff = abs(len(data) - proto_baseline_data.avg_packet_size_bytes)
        return diff <= proto_baseline_data.packet_size_std_bytes * self.std_coef


class AllDataBetweenMinMaxNetworkIO(BaseNetworkIO):
    """
    Makes sure that the data size is between the minimum and maximum packet sizes in the baseline data
    """
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        proto_baseline_data: pd.Series = self.baseline_data.loc[str(proto)]
        return proto_baseline_data.min_packet_size_bytes <= len(data) <= proto_baseline_data.max_packet_size_bytes


class NoMoreThanXPercentDeviationPerProtoNetworkIO(BaseNetworkIO):
    """
    For each protocol does not allow more than X percent of the total amount that was sent over it to be sent
    """
    def __init__(self, max_deviation_from_protos: float = .1, baseline_data: Optional[pd.DataFrame] = None):
        """
        :param max_deviation_from_protos: the maximum percent of each protocols total bytes that can be sent (10%=0.1)
        :param baseline_data: the baseline data of the communication on each protocol
        """
        super().__init__(baseline_data)
        self.max_deviation_from_protos: float = max_deviation_from_protos

        self.max_deviation_thresholds: pd.Series = pd.Series()
        self.amounts_sent_over_protos: Dict[Layer4Protocol, int] = dict()

        if baseline_data is not None:
            self.max_deviation_thresholds = self.baseline_data.total_bytes * self.max_deviation_from_protos
            self.reset()

    def reset(self):
        self.amounts_sent_over_protos = {str_to_layer4_proto(proto_str): 0 for proto_str in self.baseline_data.index}

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.max_deviation_thresholds = self.baseline_data.total_bytes * self.max_deviation_from_protos
        self.reset()

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        self.amounts_sent_over_protos[proto] += len(data)

        return self.amounts_sent_over_protos[proto] <= self.max_deviation_thresholds[str(proto)]

    def __str__(self) -> str:
        return f"NoMoreThan{self.max_deviation_from_protos * 100}PercentDeviationPerProtoNetworkIO"
