import pandas as pd
import abc

from typing import Iterable, List, Optional, Tuple

from ExfilData import ExfilData
from NetworkIO import BaseNetworkIO
from Protocols import Layer4Protocol


class BaseExfilPlanner(abc.ABC):
    """
    The base class for all exfiltration planners.
    A planner's job is to plan on which protocols to send the data,
    including the amount sent over the protocols at each action.
    Every subclass must implement the `select` method, and may need to override the `reset` method
    """

    def __init__(self, exfil_data: Optional[ExfilData] = None, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None):
        """

        :param exfil_data: the data to exfiltrate
        :param network_io: the networkIO that the planner must bypass
        :param baseline_data: the baseline data of the communication on each protocol
        """
        self.exfil_data: Optional[ExfilData] = exfil_data
        self.network_io: Optional[BaseNetworkIO] = network_io
        self.baseline_data: Optional[pd.DataFrame] = baseline_data

    def send_over_network_io(self, current_data_to_exfil: bytes, selected_proto: Optional[Layer4Protocol]) -> bool:
        """
        attempts to send the data using the protocol via the networkIO.


        :param current_data_to_exfil: the current bytes we wish to exfiltrate
        :param selected_proto: the protocol selected
        :return: if the attempt is invalid (no IO/proto), this will return False.
        otherwise, it will return the result of the sending attempt from the networkIO
        """
        if self.network_io is None:
            print(f"WARNING: no network io set for planner {self.__str__()} - returning False")
            return False

        if selected_proto is not None:
            return self.network_io.send(current_data_to_exfil, selected_proto, self.exfil_data.data_texture)
        else:
            return False

    def execute(self, return_on_first_fail: bool = False) -> List[Tuple[Optional[Layer4Protocol], bool]]:
        """
        Executes the exfiltration planned by the planner
        :param return_on_first_fail: if True, finishes the function on the first failure to send data
        :return: a list of the chosen protocols and results from the IO
        """
        if self.exfil_data is None:
            print(f"WARNING: no exfil data set for planner {self.__str__()} - returning False")
            return [(None, False)]

        action_reward_list: List[Tuple[Optional[Layer4Protocol], bool]] = list()
        for current_data_to_exfil in self.split_exfil_data():
            selected_proto: Optional[Layer4Protocol] = self.select(current_data_to_exfil)
            current_reward: bool = self.send_over_network_io(current_data_to_exfil, selected_proto)
            action_reward_list.append((selected_proto, current_reward))
            if return_on_first_fail and current_reward is False:
                break

        return action_reward_list

    def __call__(self):
        return self.execute()

    def __str__(self) -> str:
        return type(self).__name__

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data

    def set_network_io(self, network_io: BaseNetworkIO):
        self.network_io = network_io

    def set_exfil_data(self, exfil_data: ExfilData):
        self.exfil_data = exfil_data

    def split_exfil_data(self) -> Iterable[bytes]:
        return [self.exfil_data.data_to_exfiltrate]

    def reset(self):
        pass

    @abc.abstractmethod
    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        """
        selects the protocol over which the current data to exfiltrate will be sent
        :param current_data_to_exfil: the current data to exfiltrate
        :return: the chosen protocol
        """
        pass
