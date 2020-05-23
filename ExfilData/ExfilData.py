from typing import NamedTuple

from ExfilData.DataTexture import DataTextureEnum


class ExfilData(NamedTuple):
    data_to_exfiltrate: bytes
    data_texture: DataTextureEnum
