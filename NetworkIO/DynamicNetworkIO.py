import pandas as pd

from Protocols import Layer4Protocol, textual_protocols, str_to_layer4_proto
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO
from typing import Optional, Dict


class TextureNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        if proto in textual_protocols:
            proto_texture: DataTextureEnum = DataTextureEnum.textual
        else:
            proto_texture: DataTextureEnum = DataTextureEnum.binary

        return data_texture == proto_texture


class DataSizeWithinStdOfMeanForProtoNetworkIO(BaseNetworkIO):
    def __init__(self, baseline_data: Optional[pd.DataFrame] = None, std_coef: float = 1.0):
        super().__init__(baseline_data)
        self.std_coef: float = std_coef

    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        enforced_data = baseline_data.copy()

        enforced_data[baseline_data.packet_size_std_bytes == 0.0] *= 0
        return enforced_data

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        proto_baseline_data: pd.Series = self.baseline_data.loc[str(proto)]

        diff = abs(len(data) - proto_baseline_data.avg_packet_size_bytes)
        return diff <= proto_baseline_data.packet_size_std_bytes * self.std_coef


class NoMoreThanXPercentDeviationPerProtoNetworkIO(BaseNetworkIO):
    def __init__(self, max_deviation_from_protos: float = .1, baseline_data: Optional[pd.DataFrame] = None):
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
