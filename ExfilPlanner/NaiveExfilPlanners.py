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
    """
    splits a bytes array `data` to chunks of size `chunk_size` (the last one may be smaller)
    :param data: the bytes to split
    :param chunk_size: the chunk size
    :return: a generator for the sub-chunks
    """
    current_data: bytes = data
    for _ in data[::chunk_size]:
        yield current_data[:chunk_size]
        current_data = current_data[chunk_size:]


class NaiveXPercentExfilPlanner(BaseExfilPlanner):
    """
    A planner that for each protocol only sends up to X percent of the total amount sent over it in the baseline data
    """

    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None, max_deviation_from_protos: float = .1):
        """
        :param exfil_data: the data to exfiltrate
        :param network_io: the networkIO that the planner must bypass
        :param baseline_data: the baseline data of the communication on each protocol
        :param max_deviation_from_protos: the maximum percent of each protocols total bytes that we'll send. (10%=0.1)
        """
        super().__init__(exfil_data, network_io, baseline_data)
        self.max_deviation_from_protos: float = max_deviation_from_protos

        self.protocols_planned: List[Layer4Protocol] = list()
        self.amounts_left_per_protocol: List[int] = list()
        if self.baseline_data is not None:
            self.reset()

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data
        self.reset()

    def __str__(self) -> str:
        return f"Naive{self.max_deviation_from_protos * 100}PercentExfilPlanner"

    def peek_current_proto_amount(self) -> Tuple[Layer4Protocol, int]:
        """
        peeks the next protocol and amount
        :return: the next protocol and amount planned to send
        """
        return self.protocols_planned[0], self.amounts_left_per_protocol[0]

    def pop_current_proto_amount(self) -> Tuple[Layer4Protocol, int]:
        """
        pops the next protocol and amount (they are removed from the lists by this method)
        :return: the next protocol and amount planned to send
        """
        return self.protocols_planned.pop(0), self.amounts_left_per_protocol.pop(0)

    def cycle_current_proto_amount(self) -> None:
        """
        moves the current planned protocol and amount to the end of the lists
        :return: None
        """
        protocol, amount = self.pop_current_proto_amount()
        self.protocols_planned.append(protocol)
        self.amounts_left_per_protocol.append(amount)

    def take_amount_from_current_proto(self, amount: int) -> None:
        """
        removes the inputted amount from the current protcol's planned amount.
        pops the protocol and amount if there is no more data to send over the proto afterwards
        :param amount: the amount to remove
        :return: None
        """
        self.amounts_left_per_protocol[0] -= amount
        if self.amounts_left_per_protocol[0] <= 0:
            self.pop_current_proto_amount()

    def reset(self) -> None:
        """
        resets the planned amounts and protocols via the baseline data
        :return: None
        """
        self.protocols_planned = [Layer4Protocol(ProtocolEnum(proto.split(":")[0]), int(proto.split(":")[1])) for
                                  proto in self.baseline_data.index]
        self.amounts_left_per_protocol = (self.baseline_data.total_bytes * self.max_deviation_from_protos).astype(
            int).values.tolist()

    def cycle_find_sufficient_amount(self, cur_amount: int) -> bool:
        """
        cycles it's protocols and amounts until an amount >= `cur amount` is found
        :param cur_amount: the amount we want to find a protocol-amount pair for
        :return: True if a valid protocol-amount pair was found, False otherwise
        """
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
        splits by the greatest common denominator of the planned amounts
        """
        planned_amounts_gcd: int = reduce(math.gcd, self.amounts_left_per_protocol)
        return split_bytes_to_equal_chunks(self.exfil_data.data_to_exfiltrate, planned_amounts_gcd)


class NaiveSingleProtocolExfilPlanner(BaseExfilPlanner):
    """
    Only sends over the inputted protocol
    """
    def __init__(self, chosen_protocol: Layer4Protocol, exfil_data: Optional[ExfilData] = None,
                 network_io: Optional[BaseNetworkIO] = None, baseline_data: Optional[pd.DataFrame] = None):
        """
        :param chosen_protocol: the protocol over which all data will be sent
        :param exfil_data: the data to exfiltrate
        :param network_io: the networkIO that the planner must bypass
        :param baseline_data: the baseline data of the communication on each protocol
        """
        super().__init__(exfil_data, network_io, baseline_data)
        self.chosen_protocol: Layer4Protocol = chosen_protocol

    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        return self.chosen_protocol


class NaiveMaxDataProtocolExfilPlanner(NaiveSingleProtocolExfilPlanner):
    """
    Only sends over the protocol that had the most data sent over it
    """
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
    """
    Selects the protocols randomly given the weight provided
    """
    def __init__(self, weights: List[Union[int, float]], exfil_data: Optional[ExfilData] = None,
                 network_io: Optional[BaseNetworkIO] = None, baseline_data: Optional[pd.DataFrame] = None,
                 num_packets_for_split: int = 10):
        """

        :param weights: the weights for each protocol to be chosen randomly
        :param exfil_data: the data to exfiltrate
        :param network_io: the networkIO that the planner must bypass
        :param baseline_data: the baseline data of the communication on each protocol
        :param num_packets_for_split: the number of chunks to split the exfil data to
        """
        super().__init__(exfil_data, network_io, baseline_data)

        self.num_packets_for_split: int = num_packets_for_split
        if self.exfil_data is not None:
            self.set_exfil_data(exfil_data)

        self.weights: List[Union[int, float]] = weights

        self.protocols: List[Layer4Protocol] = list()
        if baseline_data is not None:
            self.protocols: List[Layer4Protocol] = [str_to_layer4_proto(proto_str) for proto_str in baseline_data.index]

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
        """
        splits to `self.num_packets_for_split` chunks
        :return: the exfil data split to `self.num_packets_for_split` chunks
        """
        packet_size: int = math.ceil(len(self.exfil_data.data_to_exfiltrate) / self.num_packets_for_split)
        return split_bytes_to_equal_chunks(self.exfil_data.data_to_exfiltrate, packet_size)


class NaiveRandomUniformExfilPlanner(NaiveRandomWeightsExfilPlanner):
    """
    Chooses from the protocols in a random & uniform way
    """
    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None, num_packets_for_split: int = 10):
        weights: List[int] = [1] * len(baseline_data.index)
        super().__init__(weights, exfil_data, network_io, baseline_data, num_packets_for_split)


class NaiveProportionalWeightsRandomExfilPlanner(NaiveRandomWeightsExfilPlanner):
    """
    Assigns for each protocol a weight proportional to the total amount sent over it
    """
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
