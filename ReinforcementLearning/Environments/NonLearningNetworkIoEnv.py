import pandas as pd
import numpy as np
import gym
from gym import spaces
import itertools
import collections

from NetworkIO import *
from Protocols import *
from ExfilData import DataTextureEnum

from typing import List, Optional, Tuple, OrderedDict, Set, Callable, ClassVar, Dict, Any


class NonLearningNetworkIoEnv(gym.Env):
    MODEL_ACTION_SPACES: ClassVar[Set[str]] = {'multidiscrete', 'discrete', 'box'}

    def __init__(self, baseline_datas: List[pd.DataFrame], network_io_fn: Callable[[], BaseNetworkIO],
                 all_protos: Set[str] = None, legal_packet_sizes: List[int] = None,
                 data_to_send_possible_values: List[int] = None, nop_penalty_size: float = 1e-2,
                 illegal_move_penalty_size: float = 1e-2, failure_penalty_size: float = 5,
                 correct_transfer_reward_factor: float = 1, model_action_space: str = 'multidiscrete',
                 add_nops: bool = False, use_random_io_mask: bool = False):
        if legal_packet_sizes is None:
            # all powers of 2 from 2^5 to 2^14
            legal_packet_sizes = [2 ** i for i in range(5, 15)]

        if all_protos is None:
            all_protos = set(itertools.chain.from_iterable(baseline_data.index for baseline_data in baseline_datas))

        self.possible_protocols: List[Optional[Layer4Protocol]] = [str_to_layer4_proto(proto_str) for proto_str in
                                                                   sorted(all_protos)]

        # NetworkIO
        self.network_io: BaseNetworkIO = network_io_fn()
        self.active_network_io: BaseNetworkIO = self.network_io
        self.num_network_ios: int = 1
        if isinstance(self.network_io, BaseEnsembleNetworkIO):
            self.network_io: BaseEnsembleNetworkIO
            self.num_network_ios = len(self.network_io.network_ios)
        self._percent_idx: int = self.__find_percent_idx()

        self.baseline_datas: List[pd.DataFrame] = self.standardized_baselines(baseline_datas, all_protos)

        if data_to_send_possible_values is None:
            data_to_send_possible_values = self.create_data_to_send_values()

        self.data_to_send_possible_values: List[int] = sorted(data_to_send_possible_values)

        # reward parameters
        self.nop_penalty_size: float = nop_penalty_size
        self.illegal_move_penalty_size: float = illegal_move_penalty_size
        self.failure_penalty_size: float = failure_penalty_size
        self.correct_transfer_reward_factor: float = correct_transfer_reward_factor
        self.legal_packet_sizes: List[Optional[int]] = sorted(legal_packet_sizes)

        if add_nops:
            # we add NOP for both protocols and amounts
            self.possible_protocols.append(None)
            self.legal_packet_sizes.append(None)

        self.action_space: spaces.Space = self.create_action_space(model_action_space)
        self.observation_space = self.create_observation_space()

        self.current_baseline_data: Optional[pd.DataFrame] = None
        self.total_data_to_send: int = 0
        self.amount_sent: int = 0
        self.amount_left: int = 0
        self.current_step: int = 0
        self.data_texture: Optional[DataTextureEnum] = None
        self.sent_values_per_proto: OrderedDict[Layer4Protocol, int] = collections.OrderedDict(
            {proto: 0 for proto in self.possible_protocols}
        )
        self.tot_rewards: int = 0
        self.moves_without_action: int = 0
        self.prev_proto_idx: int = 0
        self.prev_amount_idx: int = 0
        self.chosen_protocol_idx: int = 0
        self.chosen_amount_idx: int = 0
        self.use_random_io_mask: bool = use_random_io_mask
        self.ios_mask: List[bool] = [False] * self.num_network_ios

    @staticmethod
    def standardized_baselines(baseline_datas: List[pd.DataFrame], all_protos: Set[str]) -> List[pd.DataFrame]:
        return [
            baseline_data.append(
                pd.DataFrame(0, all_protos.difference(baseline_data.index), baseline_data.columns)).sort_index()
            for baseline_data in baseline_datas
        ]

    def create_data_to_send_values(self) -> List[int]:
        # if we have a deviation networkIO make sure to give stuff legal for it
        deviation_percent: float = self.deviation_percent()
        # if there is no networkIO for deviation, we'll say that the deviation percent is 1.0
        if not deviation_percent:
            deviation_percent = 1.0

        worst_case_enforced_baselines: List[pd.DataFrame] = [
            self.network_io.enforce_on_data(baseline_data) for baseline_data in self.baseline_datas
        ]

        max_pow_of_2_to_send: int = min(
            np.log2(baseline_data.total_bytes.sum() * deviation_percent).astype(np.int) for baseline_data in
            worst_case_enforced_baselines
        )

        # all multiples of powers of 2 that can be contained in the baselines
        return [2 ** i for i in range(10, max_pow_of_2_to_send + 1)]

    def create_action_space(self, model_action_space: str) -> spaces.Space:
        action_space_assertion_msg = f"The action space must be one of {self.MODEL_ACTION_SPACES}"
        assert model_action_space in NonLearningNetworkIoEnv.MODEL_ACTION_SPACES, action_space_assertion_msg

        action_space: Optional[spaces.Space] = None

        if model_action_space == 'box':
            # using box because not all algorithms support MultiDiscrete, we'll round the values to discrete actions
            action_space = spaces.Box(
                low=np.array([0, 0]),
                high=np.array([len(self.possible_protocols), len(self.legal_packet_sizes)]) - np.finfo(np.float32).eps,
                dtype=np.int64
            )
        elif model_action_space == 'discrete':
            action_space = spaces.Discrete(len(self.possible_protocols) * len(self.legal_packet_sizes))
        elif model_action_space == 'multidiscrete':
            action_space = spaces.MultiDiscrete(
                [len(self.possible_protocols), len(self.legal_packet_sizes)]
            )

        assert action_space is not None
        return action_space

    def create_observation_space(self) -> spaces.Space:
        num_protocols_values = len(self.possible_protocols) * len(self.baseline_datas[0].columns)
        # total_data_to_send, amount_sent, amount_left, data texture, statistics per protocol (num_columns * num_rows)
        # amount sent per proto (num_protos), last chosen protocol, last chosen amount
        observation_space = spaces.Box(
            low=np.array(
                [min(self.data_to_send_possible_values)] + [0] * 2 + [0] + [0] * num_protocols_values + [0] * len(
                    self.possible_protocols) + [0] * 2
            ),
            high=np.array(
                [max(self.data_to_send_possible_values)] * 3 + [1] + [np.inf] * num_protocols_values + [
                    max(self.data_to_send_possible_values)] * len(self.possible_protocols) + [
                    len(self.possible_protocols), len(self.legal_packet_sizes)]
            )
        )

        return observation_space

    def __find_percent_idx(self) -> int:
        if isinstance(self.network_io, BaseEnsembleNetworkIO):
            self.network_io: BaseEnsembleNetworkIO
            for idx, network_io in enumerate(self.network_io.network_ios):
                if isinstance(network_io, NoMoreThanXPercentDeviationPerProtoNetworkIO):
                    return idx
        elif isinstance(self.network_io, NoMoreThanXPercentDeviationPerProtoNetworkIO):
            self.network_io: NoMoreThanXPercentDeviationPerProtoNetworkIO
            return 0

        return -1

    def deviation_percent(self) -> Optional[float]:
        if isinstance(self.network_io, BaseEnsembleNetworkIO) and self._percent_idx >= 0:
            self.network_io: BaseEnsembleNetworkIO
            return self.network_io.network_ios[self._percent_idx].max_deviation_from_protos
        elif self._percent_idx == 0:
            self.network_io: NoMoreThanXPercentDeviationPerProtoNetworkIO
            return self.network_io.max_deviation_from_protos

        return None

    def get_current_state_observation(self) -> np.ndarray:
        obs = np.append(
            [self.total_data_to_send, self.amount_sent, self.amount_left, self.data_texture.value],
            self.current_baseline_data.to_numpy().reshape(-1)
        )
        obs = np.append(
            obs, list(self.sent_values_per_proto.values())
        )
        obs = np.append(
            obs, [
                self.prev_proto_idx,
                self.prev_amount_idx,
            ]
        )

        return obs

    def mask_network_ios(self) -> None:
        if self.num_network_ios == 1:
            self.ios_mask = [True]
        else:
            self.network_io: BaseEnsembleNetworkIO
            self.ios_mask = [False] * self.num_network_ios

            while not any(self.ios_mask):
                self.ios_mask = np.random.choice([True, False], self.num_network_ios).tolist()

            if self._percent_idx >= 0:
                self.ios_mask[self._percent_idx] = True

            self.active_network_io: BaseEnsembleNetworkIO = self.network_io.ios_subset(self.ios_mask)

    def reset(self) -> np.ndarray:
        # set the state randomly
        self.current_step = 0
        self.data_texture = np.random.choice(DataTextureEnum)
        self.current_baseline_data = self.baseline_datas[np.random.choice(range(len(self.baseline_datas)))].copy()

        # reset the metadata
        self.tot_rewards = 0
        self.moves_without_action = 0
        self.prev_proto_idx = len(self.possible_protocols)
        self.prev_amount_idx = len(self.legal_packet_sizes)
        self.chosen_protocol_idx = 0
        self.chosen_amount_idx = 0

        # if needed, choose a subset of network ios to use
        if self.use_random_io_mask:
            self.mask_network_ios()

        # enforce the network io on the data
        self.current_baseline_data = self.active_network_io.enforce_on_data(self.current_baseline_data)

        # reset sending amounts
        send_value_threshold: float = np.inf
        if self.deviation_percent():
            send_value_threshold = self.current_baseline_data.total_bytes.sum() * self.deviation_percent()

        self.total_data_to_send = np.random.choice(
            [val for val in self.data_to_send_possible_values if val <= send_value_threshold]
        )
        self.amount_left = self.total_data_to_send
        self.amount_sent = 0
        self.sent_values_per_proto: OrderedDict[Layer4Protocol, int] = collections.OrderedDict(
            {proto: 0 for proto in self.possible_protocols}
        )

        # reset the networkIo
        self.active_network_io.set_baseline_data(self.current_baseline_data)
        self.active_network_io.reset()

        return self.get_current_state_observation()

    def render(self, mode='human'):
        print(f"step:\t\t\t{self.current_step}")
        print(f"total data:\t\t{self.total_data_to_send}")
        print(f"data sent:\t\t{self.amount_sent}")
        print(f"data left:\t\t{self.amount_left}")
        print(f"data texture:\t{self.data_texture.name}")

    def is_action_legal(self, chosen_protocol: Optional[Layer4Protocol], chosen_amount: Optional[int]) -> bool:
        if chosen_protocol is None and chosen_amount is None:
            return True

        elif chosen_protocol is None or chosen_amount is None:
            return False

        return chosen_amount <= self.amount_left

    def transfer_reward(self, amount_sent: int) -> float:
        return self.correct_transfer_reward_factor * (amount_sent / self.total_data_to_send)

    def action_to_proto_amount(self, action: np.ndarray) -> Tuple[Optional[Layer4Protocol], int]:
        if isinstance(self.action_space, spaces.Box):
            action = action.astype(np.int64)
        elif isinstance(self.action_space, spaces.Discrete):
            action = np.array(np.divmod(action, np.array([len(self.legal_packet_sizes)]))).reshape(-1)

        chosen_protocol_idx, chosen_amount_idx = action[0], action[1]

        self.chosen_protocol_idx = min(chosen_protocol_idx, len(self.possible_protocols) - 1)
        self.chosen_amount_idx = min(chosen_amount_idx, len(self.legal_packet_sizes) - 1)

        return self.possible_protocols[self.chosen_protocol_idx], self.legal_packet_sizes[self.chosen_amount_idx]

    def step_result(self, reward: float, done: bool = False,
                    info: Dict[str, Any] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if info is None:
            info = dict()

        self.tot_rewards += reward

        self.prev_proto_idx, self.prev_amount_idx = self.chosen_protocol_idx, self.chosen_amount_idx

        if self.tot_rewards < 0 and self.current_step > 1000 or self.moves_without_action > 20:
            info.update(dict(success=False))
            return self.get_current_state_observation(), -self.failure_penalty_size, True, info

        return self.get_current_state_observation(), reward, done, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        chosen_protocol, chosen_amount = self.action_to_proto_amount(action)

        self.current_step += 1

        if not self.is_action_legal(chosen_protocol, chosen_amount):
            self.moves_without_action += 1
            return self.step_result(-self.illegal_move_penalty_size,
                                    info=dict(moves_without_action=self.moves_without_action))

        if chosen_protocol is None and chosen_amount is None:
            self.moves_without_action += 1
            return self.step_result(-self.nop_penalty_size, info=dict(moves_without_action=self.moves_without_action))

        send_res = self.active_network_io.send(bytes(chosen_amount), chosen_protocol, self.data_texture)

        if send_res:
            self.amount_left -= chosen_amount
            self.amount_sent += chosen_amount
            self.sent_values_per_proto[chosen_protocol] += chosen_amount

            done = (self.amount_left == 0)
            if done:
                info = dict(success=True)
            else:
                info = None

            return self.step_result(self.transfer_reward(chosen_amount), done, info)
        else:
            return self.step_result(-self.failure_penalty_size, True, dict(success=False))
