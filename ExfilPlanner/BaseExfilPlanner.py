import pandas as pd
import abc

from typing import Iterable, List, Optional, Tuple

from ExfilData import ExfilData
from NetworkIO import BaseNetworkIO
from Protocols import Layer4Protocol


class BaseExfilPlanner(abc.ABC):
    def __init__(self, exfil_data: ExfilData, network_io: Optional[BaseNetworkIO] = None,
                 baseline_data: Optional[pd.DataFrame] = None):
        self.exfil_data: ExfilData = exfil_data
        self.network_io: Optional[BaseNetworkIO] = network_io
        self.baseline_data: Optional[pd.DataFrame] = baseline_data

    def send_over_network_io(self, current_data_to_exfil: bytes, selected_proto: Optional[Layer4Protocol]) -> bool:
        if self.network_io is None:
            print(f"WARNING: no network io set for planner {self.__str__()} - returning False")
            return False

        if selected_proto is not None:
            return self.network_io.send(current_data_to_exfil, selected_proto, self.exfil_data.data_texture)
        else:
            return False

    def execute(self) -> List[Tuple[Optional[Layer4Protocol], bool]]:
        action_reward_list: List[Tuple[Optional[Layer4Protocol], bool]] = list()
        for current_data_to_exfil in self.split_exfil_data():
            selected_proto: Optional[Layer4Protocol] = self.select(current_data_to_exfil)
            current_reward: bool = self.send_over_network_io(current_data_to_exfil, selected_proto)
            action_reward_list.append((selected_proto, current_reward))

        return action_reward_list

    def __call__(self):
        return self.execute()

    def __str__(self) -> str:
        return type(self).__name__

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data

    def set_network_io(self, network_io: BaseNetworkIO):
        self.network_io = network_io

    def split_exfil_data(self) -> Iterable[bytes]:
        return [self.exfil_data.data_to_exfiltrate]

    @abc.abstractmethod
    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        pass
