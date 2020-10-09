from enum import Enum
from typing import NamedTuple


class ProtocolEnum(Enum):
    """
    An enum describing the layer 3 base protocols
    """
    TCP = "TCP"
    UDP = "UDP"


class Layer4Protocol(NamedTuple):
    """
    Describes a layer 4 (app) protocol with it's base protocol and destination port
    """
    layer3_proto: ProtocolEnum
    dst_port: int

    def __str__(self) -> str:
        return f"{self.layer3_proto.value}:{self.dst_port}"


def str_to_layer4_proto(proto_str: str) -> Layer4Protocol:
    """
    creates a Layer4Protocol from a string with the format "<TCP|UDP>:<dst_port>"
    :param proto_str: the string describing the protocol
    :return: a `Layer4Protocol` object matching the inputted string
    """
    layer3_proto_str, dst_port_str = proto_str.split(":")
    return Layer4Protocol(ProtocolEnum(layer3_proto_str), int(dst_port_str))
