import pandas as pd
import random
import math
from functools import reduce

from typing import Iterable, List, Tuple, Optional, Union

from ExfilData import ExfilData
from NetworkIO import BaseNetworkIO
from Protocols import Layer4Protocol, ProtocolEnum, str_to_layer4_proto
from ExfilPlanner.BaseExfilPlanner import BaseExfilPlanner


def split_bytes_to_equal_chunks(data: bytes, chunk_size: int) -> Iterable[bytes]:
    current_data: bytes = data
    for _ in data[::chunk_size]:
        yield current_data[:chunk_size]
        current_data = current_data[chunk_size:]


class NaiveXPercentExfilPlanner(BaseExfilPlanner):
    def __init__(self, exfil_data: ExfilData, network_io: BaseNetworkIO, baseline_data: Optional[pd.DataFrame] = None,
                 max_deviation_from_protos: float = .1):
        super().__init__(exfil_data, network_io, baseline_data)
        self.max_deviation_from_protos: float = max_deviation_from_protos

        self.__protocols_planned: List[Layer4Protocol] = list()
        self.__amounts_left_per_protocol: List[int] = list()
        if self.baseline_data is not None:
            self.__plan_amounts()

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.__plan_amounts()

    def __str__(self) -> str:
        return f"Naive{self.max_deviation_from_protos * 100}PercentExfilPlanner"

    def __peek_current_proto_amount(self) -> Tuple[Layer4Protocol, int]:
        return self.__protocols_planned[0], self.__amounts_left_per_protocol[0]

    def __pop_current_proto_amount(self) -> Tuple[Layer4Protocol, int]:
        return self.__protocols_planned.pop(0), self.__amounts_left_per_protocol.pop(0)

    def __cycle_current_proto_amount(self):
        protocol, amount = self.__pop_current_proto_amount()
        self.__protocols_planned.append(protocol)
        self.__amounts_left_per_protocol.append(amount)

    def __take_amount_from_current_proto(self, amount: int):
        self.__amounts_left_per_protocol[0] -= amount
        if self.__amounts_left_per_protocol[0] <= 0:
            self.__pop_current_proto_amount()

    def __plan_amounts(self):
        self.__protocols_planned = [Layer4Protocol(ProtocolEnum(proto.split(":")[0]), int(proto.split(":")[1])) for
                                    proto in self.baseline_data.index]
        self.__amounts_left_per_protocol = (self.baseline_data['total_bytes'] * self.max_deviation_from_protos).astype(
            int).values.tolist()

    def __cycle_find_sufficient_amount(self, cur_amount: int) -> bool:
        for _ in range(len(self.__amounts_left_per_protocol)):
            if self.__peek_current_proto_amount()[1] >= cur_amount:
                return True
            else:
                self.__cycle_current_proto_amount()
        return False

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        cur_amount = len(current_data_to_exfil)
        if self.__cycle_find_sufficient_amount(cur_amount):
            selected_proto: Layer4Protocol = self.__peek_current_proto_amount()[0]
            self.__take_amount_from_current_proto(cur_amount)
            return selected_proto
        else:
            return None

    def split_exfil_data(self) -> Iterable[bytes]:
        """
        split by the greatest common denominator of the planned amounts
        """
        planned_amounts_gcd: int = reduce(math.gcd, self.__amounts_left_per_protocol)
        return split_bytes_to_equal_chunks(self.exfil_data.data_to_exfiltrate, planned_amounts_gcd)


class NaiveSingleProtocolExfilPlanner(BaseExfilPlanner):
    def __init__(self, exfil_data: ExfilData, network_io: BaseNetworkIO, chosen_protocol: Layer4Protocol,
                 baseline_data: Optional[pd.DataFrame] = None):
        super().__init__(exfil_data, network_io, baseline_data)
        self.chosen_protocol: Layer4Protocol = chosen_protocol

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        return self.chosen_protocol


class NaiveMaxDataProtocolExfilPlanner(NaiveSingleProtocolExfilPlanner):
    def __init__(self, exfil_data: ExfilData, network_io: BaseNetworkIO, baseline_data: Optional[pd.DataFrame] = None):
        super().__init__(exfil_data, network_io, str_to_layer4_proto(baseline_data.total_bytes.idxmax()), baseline_data)


class NaiveRandomWeightsExfilPlanner(BaseExfilPlanner):
    def __init__(self, exfil_data: ExfilData, network_io: BaseNetworkIO, weights: List[Union[int, float]],
                 baseline_data: Optional[pd.DataFrame] = None, num_packets_for_split: int = 10):
        super().__init__(exfil_data, network_io, baseline_data)

        num_packets_for_split = min(num_packets_for_split, len(exfil_data.data_to_exfiltrate))
        self.num_packets_for_split: int = num_packets_for_split

        self.weights: List[Union[int, float]] = weights
        self.protocols: List[Layer4Protocol] = [str_to_layer4_proto(proto_str) for proto_str in
                                                self.baseline_data.index]

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        return random.choices(self.protocols, weights=self.weights)[0]

    def split_exfil_data(self) -> Iterable[bytes]:
        packet_size: int = math.ceil(len(self.exfil_data.data_to_exfiltrate) / self.num_packets_for_split)
        return split_bytes_to_equal_chunks(self.exfil_data.data_to_exfiltrate, packet_size)


class NaiveRandomUniformExfilPlanner(NaiveRandomWeightsExfilPlanner):
    def __init__(self, exfil_data: ExfilData, network_io: BaseNetworkIO, baseline_data: Optional[pd.DataFrame] = None):
        weights: List[int] = [1 for _ in range(len(baseline_data.index))]
        super().__init__(exfil_data, network_io, baseline_data=baseline_data, weights=weights)


class NaiveProportionalWeightsRandomExfilPlanner(NaiveRandomWeightsExfilPlanner):
    def __init__(self, exfil_data: ExfilData, network_io: BaseNetworkIO, baseline_data: Optional[pd.DataFrame] = None):
        weights: List[Union[int, float]] = baseline_data.total_bytes.values.tolist()
        super().__init__(exfil_data, network_io, baseline_data=baseline_data, weights=weights)
