import pandas as pd
import abc

from typing import Iterable, List, Optional, Tuple

from NetworkIO import BaseNetworkIO
from Protocols import Layer4Protocol


class BaseExfilPlanner(abc.ABC):
    def __init__(self, exfil_data: bytes, network_io: BaseNetworkIO, baseline_data: pd.DataFrame):
        self.exfil_data: bytes = exfil_data
        self.network_io: BaseNetworkIO = network_io
        self.baseline_data: pd.DataFrame = baseline_data

    def send_over_network_io(self, current_data_to_exfil: bytes, selected_proto: Optional[Layer4Protocol]) -> bool:
        if selected_proto is not None:
            return self.network_io.send(current_data_to_exfil, selected_proto)
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

    @abc.abstractmethod
    def split_exfil_data(self) -> Iterable[bytes]:
        return [self.exfil_data]

    @abc.abstractmethod
    def select(self, current_data_to_exfil: bytes) -> Optional[Layer4Protocol]:
        pass
