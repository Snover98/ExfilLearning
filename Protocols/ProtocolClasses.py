from enum import Enum
from typing import NamedTuple


class ProtocolEnum(Enum):
    TCP = "TCP"
    UDP = "UDP"


class Layer4Protocol(NamedTuple):
    layer3_proto: ProtocolEnum
    dst_port: int


def str_to_layer4_proto(proto_str: str) -> Layer4Protocol:
    layer3_proto_str, dst_port_str = proto_str.split(":")
    return Layer4Protocol(ProtocolEnum(layer3_proto_str), int(dst_port_str))
