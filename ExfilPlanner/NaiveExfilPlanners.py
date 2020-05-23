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
    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None, max_deviation_from_protos: float = .1):
        super().__init__(exfil_data, network_io, baseline_data)
        self.max_deviation_from_protos: float = max_deviation_from_protos

        self.protocols_planned: List[Layer4Protocol] = list()
        self.amounts_left_per_protocol: List[int] = list()
        if self.baseline_data is not None:
            self.plan_amounts()

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.plan_amounts()

    def __str__(self) -> str:
        return f"Naive{self.max_deviation_from_protos * 100}PercentExfilPlanner"

    def peek_current_proto_amount(self) -> Tuple[Layer4Protocol, int]:
        return self.protocols_planned[0], self.amounts_left_per_protocol[0]

    def pop_current_proto_amount(self) -> Tuple[Layer4Protocol, int]:
        return self.protocols_planned.pop(0), self.amounts_left_per_protocol.pop(0)

    def cycle_current_proto_amount(self):
        protocol, amount = self.pop_current_proto_amount()
        self.protocols_planned.append(protocol)
        self.amounts_left_per_protocol.append(amount)

    def take_amount_from_current_proto(self, amount: int):
        self.amounts_left_per_protocol[0] -= amount
        if self.amounts_left_per_protocol[0] <= 0:
            self.pop_current_proto_amount()

    def plan_amounts(self):
        self.protocols_planned = [Layer4Protocol(ProtocolEnum(proto.split(":")[0]), int(proto.split(":")[1])) for
                                  proto in self.baseline_data.index]
        self.amounts_left_per_protocol = (self.baseline_data['total_bytes'] * self.max_deviation_from_protos).astype(
            int).values.tolist()

    def cycle_find_sufficient_amount(self, cur_amount: int) -> bool:
        for _ in range(len(self.amounts_left_per_protocol)):
            if self.peek_current_proto_amount()[1] >= cur_amount:
                return True
            else:
                self.cycle_current_proto_amount()
        return False

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        cur_amount = len(current_data_to_exfil)
        if self.cycle_find_sufficient_amount(cur_amount):
            selected_proto: Layer4Protocol = self.peek_current_proto_amount()[0]
            self.take_amount_from_current_proto(cur_amount)
            return selected_proto
        else:
            return None

    def split_exfil_data(self) -> Iterable[bytes]:
        """
        split by the greatest common denominator of the planned amounts
        """
        planned_amounts_gcd: int = reduce(math.gcd, self.amounts_left_per_protocol)
        return split_bytes_to_equal_chunks(self.exfil_data.data_to_exfiltrate, planned_amounts_gcd)

    def plan(self):
        self.plan_amounts()


class NaiveSingleProtocolExfilPlanner(BaseExfilPlanner):
    def __init__(self, chosen_protocol: Layer4Protocol, exfil_data: Optional[ExfilData] = None,
                 network_io: Optional[BaseNetworkIO] = None, baseline_data: Optional[pd.DataFrame] = None):
        super().__init__(exfil_data, network_io, baseline_data)
        self.chosen_protocol: Layer4Protocol = chosen_protocol

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        return self.chosen_protocol


class NaiveMaxDataProtocolExfilPlanner(NaiveSingleProtocolExfilPlanner):
    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None):
        if baseline_data is None:
            max_proto: Optional[Layer4Protocol] = None
        else:
            max_proto: Optional[Layer4Protocol] = str_to_layer4_proto(baseline_data.total_bytes.idxmax())
        super().__init__(max_proto, exfil_data, network_io, baseline_data)

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.chosen_protocol = str_to_layer4_proto(baseline_data.total_bytes.idxmax())


class NaiveRandomWeightsExfilPlanner(BaseExfilPlanner):
    def __init__(self, weights: List[Union[int, float]], exfil_data: Optional[ExfilData] = None,
                 network_io: Optional[BaseNetworkIO] = None, baseline_data: Optional[pd.DataFrame] = None,
                 num_packets_for_split: int = 10):
        super().__init__(exfil_data, network_io, baseline_data)

        if self.exfil_data is not None:
            num_packets_for_split = min(num_packets_for_split, len(exfil_data.data_to_exfiltrate))
        self.num_packets_for_split: int = num_packets_for_split

        self.weights: List[Union[int, float]] = weights
        if baseline_data is not None:
            self.protocols: List[Layer4Protocol] = [str_to_layer4_proto(proto_str) for proto_str in baseline_data.index]
        else:
            self.protocols: List[Layer4Protocol] = list()

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.protocols: List[Layer4Protocol] = [str_to_layer4_proto(proto_str) for proto_str in baseline_data.index]

    def set_exfil_data(self, exfil_data: ExfilData):
        self.exfil_data = exfil_data
        self.num_packets_for_split: int = min(self.num_packets_for_split, len(exfil_data.data_to_exfiltrate))

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        if self.baseline_data is None:
            print(f"WARNING: no baseline data set for planner {self.__str__()} - returning None")
            return None

        return random.choices(self.protocols, weights=self.weights)[0]

    def split_exfil_data(self) -> Iterable[bytes]:
        packet_size: int = math.ceil(len(self.exfil_data.data_to_exfiltrate) / self.num_packets_for_split)
        return split_bytes_to_equal_chunks(self.exfil_data.data_to_exfiltrate, packet_size)


class NaiveRandomUniformExfilPlanner(NaiveRandomWeightsExfilPlanner):
    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None, num_packets_for_split: int = 10):
        weights: List[int] = [1 for _ in range(len(baseline_data.index))]
        super().__init__(weights, exfil_data, network_io, baseline_data, num_packets_for_split)


class NaiveProportionalWeightsRandomExfilPlanner(NaiveRandomWeightsExfilPlanner):
    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None, num_packets_for_split: int = 10):
        if baseline_data is not None:
            weights: List[Union[int, float]] = baseline_data.total_bytes.values.tolist()
        else:
            weights: List[Union[int, float]] = list()
        super().__init__(weights, exfil_data, network_io, baseline_data, num_packets_for_split)

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        super().set_baseline_data(baseline_data)
        self.weights: List[Union[int, float]] = baseline_data.total_bytes.values.tolist()
