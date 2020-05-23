import pandas as pd

from Protocols import Layer4Protocol, textual_protocols
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO


class TextureNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        if proto in textual_protocols:
            proto_texture: DataTextureEnum = DataTextureEnum.textual
        else:
            proto_texture: DataTextureEnum = DataTextureEnum.binary

        return data_texture == proto_texture


class DataSizeWithinStdOfMeanForProtoNetworkIO(BaseNetworkIO):
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        proto_baseline_data: pd.Series = self.baseline_data.loc[str(proto)]

        diff = abs(len(data) - proto_baseline_data.avg_packet_size_bytes)
        return diff <= proto_baseline_data.packet_size_std_bytes
