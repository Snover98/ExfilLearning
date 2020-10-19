import pandas as pd
import numpy as np

import abc

from Protocols import Layer4Protocol
from ExfilData import DataTextureEnum
from NetworkIO.BaseNetworkIO import BaseNetworkIO

from typing import Optional, List, Tuple


class BaseEnsembleNetworkIO(BaseNetworkIO):
    """
    A base class for a NetworkIO containing other NetworkIOs
    """
    def __init__(self, network_ios: List[BaseNetworkIO], baseline_data: Optional[pd.DataFrame] = None):
        """
        :param network_ios: a list of the ensembles' NetworkIOs
        :param baseline_data: the baseline data of the communication on each protocol
        """
        super().__init__(baseline_data)
        self.network_ios: List[BaseNetworkIO] = network_ios

    def mask_ios(self, mask: List[bool]) -> List[BaseNetworkIO]:
        """
        Masks the network ios by a list of booleans - returns the ones whose positions had a value of `True`

        :param mask: a list of booleans to mask the NetworkIOs
        :return: a list of the networkIOs that had the masking value of `True`
        """
        assert len(mask) == len(self.network_ios), "Mask must be as long as number of network ios"
        assert any(mask), "At least one network io must be enabled in the mask"
        return [network_io for network_io, mask in zip(self.network_ios, mask) if mask]

    def ios_subset(self, mask: List[bool]) -> 'BaseEnsembleNetworkIO':
        """
        returns a new ensemble of the same type with only the network ios that had `True` in their mask
        :param mask: a list of booleans, a value of `True` means that the io of the same index will be in the subset
        :return: a new ensemble of the same type with only the network ios that had `True` in their mask
        """
        masked_ios = self.mask_ios(mask)
        return type(self)(masked_ios, self.baseline_data)

    def calc_network_ios_decisions(self, data: bytes, proto: Layer4Protocol,
                                   data_texture: DataTextureEnum) -> Tuple[bool, ...]:
        """
        calculates the decision of each network io in the ensemble over the `send` parameters
        :param data: the data to send in bytes
        :param proto: the protocol over which the data will be sent
        :param data_texture: the data's texture
        :return: a tuple of the NetworkIOs decisions
        """
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
    """
    Lets data be sent only if all of it's NetworkIOs agreed that it should be sent
    """
    def enforce_on_data(self, baseline_data: pd.DataFrame) -> pd.DataFrame:
        """
        enforces all of the contained NetworkIOs on the baseline data
        """
        enforced_data = baseline_data.copy()

        for network_io in self.network_ios:
            enforced_data = network_io.enforce_on_data(enforced_data)

        return enforced_data

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        return all(self.calc_network_ios_decisions(data, proto, data_texture))


class VotingEnsembleNetworkIO(BaseEnsembleNetworkIO):
    """
    Decides the send result via a vote between it's NetworkIos
    Each NetworkIO has a voting weight (not all voters must have the same weight)
    """
    def __init__(self, network_ios: List[BaseNetworkIO], voting_weights: Optional[List[float]] = None,
                 baseline_data: Optional[pd.DataFrame] = None, tie_breaker_result: Optional[bool] = True):
        """
        :param network_ios: a list of the ensembles' NetworkIOs
        :param voting_weights: the weight of each IO. defaults to 1/num_of_ios each and normalized to have a sum of 1
        :param baseline_data: the baseline data of the communication on each protocol
        :param tie_breaker_result: the value to return in the case of a voting tie. defaults to `True`
        """
        super().__init__(network_ios, baseline_data)
        self.tie_breaker_result: bool = tie_breaker_result

        if voting_weights is None:
            voting_weights = [1 / len(network_ios)] * len(network_ios)

        if len(voting_weights) != len(network_ios):
            exception_msg = "Invalid init values for VotingEnsembleNetworkIO, len(voting_weights) != len(network_ios)"
            exception_msg = f"{exception_msg}  ({len(voting_weights)} != {len(network_ios)})"
            raise ValueError(exception_msg)

        self.voting_weights = np.array(voting_weights)
        # normalize to make the sum 1
        self.voting_weights = self.voting_weights / self.voting_weights.sum()

    def ios_subset(self, mask: List[bool]) -> 'BaseEnsembleNetworkIO':
        """
        Implemented here because the constructor has different params
        """
        masked_ios = self.mask_ios(mask)
        return type(self)(masked_ios, self.voting_weights, self.baseline_data, self.tie_breaker_result)

    def send(self, data: bytes, proto: Layer4Protocol, data_texture: DataTextureEnum) -> bool:
        network_ios_votes: Tuple[bool, ...] = self.calc_network_ios_decisions(data, proto, data_texture)

        vote_values = np.array([int(vote) for vote in network_ios_votes])
        weighted_votes_sum: float = vote_values @ self.voting_weights

        if weighted_votes_sum == 0.5:
            return self.tie_breaker_result

        return weighted_votes_sum > 0.5



