import pandas as pd
import numpy as np

import abc

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO

from typing import Optional, List, Tuple


class BaseEnsembleNetworkIO(BaseNetworkIO):
    def __init__(self, network_ios: List[BaseNetworkIO], baseline_data: Optional[pd.DataFrame] = None):
        super().__init__(baseline_data)
        self.network_ios: List[BaseNetworkIO] = network_ios

    def mask_ios(self, mask: List[bool]) -> List[BaseNetworkIO]:
        assert len(mask) == len(self.network_ios), "Mask must be as long as number of network ios"
        assert any(mask), "At least one network io must be enabled in the mask"
        return [self.network_ios[idx] for idx in range(len(mask)) if mask[idx]]

    def ios_subset(self, mask: List[bool]) -> 'BaseEnsembleNetworkIO':
        masked_ios = self.mask_ios(mask)
        return type(self)(masked_ios, self.baseline_data)

    def calc_network_ios_decisions(self, data: bytes, proto: Layer4Protocol,
                                   data_texture: DataTextureEnum) -> Tuple[bool, ...]:
        return tuple([network_io.send(data, proto, data_texture) for network_io in self.network_ios])

    def set_baseline_data(self, baseline_data: pd.DataFrame):
        self.baseline_data = baseline_data

        for network_io in self.network_ios:
            network_io.set_baseline_data(baseline_data)

    def __str__(self) -> str:
        network_ios_names: List[str] = [str(network_io) for network_io in self.network_ios]
        return f"{type(self).__name__}({','.join(network_ios_names)})"

    @abc.abstractmethod
    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        pass

    def reset(self):
        for network_io in self.network_ios:
            network_io.reset()


class FullConsensusEnsembleNetworkIO(BaseEnsembleNetworkIO):
    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        enforced_data = baseline_data.copy()

        for network_io in self.network_ios:
            enforced_data = network_io.enforce_on_data(enforced_data)

        return enforced_data

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return all(self.calc_network_ios_decisions(data, proto, data_texture))


class VotingEnsembleNetworkIO(BaseEnsembleNetworkIO):
    def __init__(self, network_ios: List[BaseNetworkIO], voting_weights: Optional[List[float]] = None,
                 baseline_data: Optional[pd.DataFrame] = None, tie_breaker_result: Optional[bool] = True):
        super().__init__(network_ios, baseline_data)
        self.tie_breaker_result: bool = tie_breaker_result

        if voting_weights is None:
            voting_weights = [1 / len(network_ios)] * len(network_ios)

        if len(voting_weights) != len(network_ios):
            exception_msg = "Invalid init values for VotingEnsembleNetworkIO, len(voting_weights) != len(network_ios)"
            exception_msg = f"{exception_msg}  ({len(voting_weights)} != {len(network_ios)})"
            raise Exception(exception_msg)

        self.voting_weights = np.array(voting_weights)
        # normalize to make the sum 1
        self.voting_weights = self.voting_weights / self.voting_weights.sum()

    def ios_subset(self, mask: List[bool]) -> 'BaseEnsembleNetworkIO':
        masked_ios = self.mask_ios(mask)
        return type(self)(masked_ios, self.voting_weights, self.baseline_data, self.tie_breaker_result)

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        network_ios_votes: Tuple[bool, ...] = self.calc_network_ios_decisions(data, proto, data_texture)

        vote_values = np.array([int(vote) for vote in network_ios_votes])
        weighted_votes_sum: float = vote_values @ self.voting_weights

        if weighted_votes_sum == 0.5:
            return self.tie_breaker_result

        return weighted_votes_sum > 0.5



