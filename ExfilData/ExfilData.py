from typing import NamedTuple

from .DataTexture import DataTextureEnum


class ExfilData(NamedTuple):
    data_to_exfiltrate: bytes
    data_texture: DataTextureEnum
