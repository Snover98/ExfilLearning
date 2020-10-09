from typing import NamedTuple

from ExfilData.DataTexture import DataTextureEnum


class ExfilData(NamedTuple):
    """
    A tuple describing data to exfiltrate, contains the data bytes and the data texture
    """
    data_to_exfiltrate: bytes
    data_texture: DataTextureEnum
