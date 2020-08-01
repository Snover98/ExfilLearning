import pandas as pd
import numpy as np
import gym
from gym import spaces
import itertools
import collections

from NetworkIO import *
from Protocols import *
from ExfilData import DataTextureEnum

from typing import List, Optional, Tuple, OrderedDict, Set, Callable


class NonLearningNetworkIoEnv(gym.Env):
    def __init__(self, baseline_datas: List[pd.DataFrame], network_io_fn: Callable[[], BaseNetworkIO],
                 all_protos: Set[str] = None, legal_packet_sizes: List[int] = None,
                 data_to_send_possible_values: List[int] = None, nop_penalty_size: float = 1e-2,
                 illegal_move_penalty_size: float = 1e-2, failure_penalty_size: float = 5,
                 correct_transfer_reward_factor: float = 1, use_box: bool = False, add_nops: bool = False):

        if legal_packet_sizes is None:
            # all powers of 2 from 2^5 to 2^14
            legal_packet_sizes = [2 ** i for i in range(5, 15)]

        if data_to_send_possible_values is None:
            # all multiples of powers of 2 from 2^10 to 2^20
            data_to_send_possible_values = list(
                itertools.starmap(lambda x, y: x * y,
                                  itertools.product([2 ** i for i in range(10, 21)], range(1, 10, 2)))
            )

        if all_protos is None:
            protos_str_per_baseline = [baseline_data.index for baseline_data in baseline_datas]
            all_protos = set(itertools.chain.from_iterable(protos_str_per_baseline))

        self.possible_protocols: List[Optional[Layer4Protocol]] = [str_to_layer4_proto(proto_str) for proto_str in
                                                                   sorted(all_protos)]

        baseline_datas = [
            baseline_data.append(
                pd.DataFrame(0, all_protos.difference(baseline_data.index), baseline_data.columns)).sort_index()
            for baseline_data in baseline_datas
        ]

        self.baseline_datas: List[pd.DataFrame] = baseline_datas

        self.network_io_fn: Callable[[], BaseNetworkIO] = network_io_fn
        self.network_io: BaseNetworkIO = network_io_fn()
        self.data_to_send_possible_values: List[int] = data_to_send_possible_values

        self.nop_penalty_size: float = nop_penalty_size
        self.illegal_move_penalty_size: float = illegal_move_penalty_size
        self.failure_penalty_size: float = failure_penalty_size
        self.correct_transfer_reward_factor: float = correct_transfer_reward_factor

        self.legal_packet_sizes: List[Optional[int]] = legal_packet_sizes.copy()

        if add_nops:
            # we add NOP in each discrete action space
            self.possible_protocols.append(None)
            self.legal_packet_sizes.append(None)

        if use_box:
            # using box because not all algorithms support MultiDiscrete, we'll round the values to discrete actions
            self.action_space: spaces.Space = spaces.Box(
                low=np.array([0, 0]),
                high=np.array([len(self.possible_protocols), len(self.legal_packet_sizes)]),
                dtype=np.int64
            )
        else:
            self.action_space: spaces.Space = spaces.MultiDiscrete(
                [len(self.possible_protocols), len(self.legal_packet_sizes)]
            )

        # total_data_to_send, amount_sent, amount_left, data texture, statistics per protocol (num_columns * num_rows)
        num_protocols_values = len(self.possible_protocols) * len(self.baseline_datas[0].columns)
        self.observation_space = spaces.Box(
            low=np.array([min(data_to_send_possible_values)] * 3 + [0] + [0] * num_protocols_values + [0] * len(
                self.possible_protocols)),
            high=np.array(
                [max(data_to_send_possible_values)] * 3 + [1] + [np.inf] * num_protocols_values + [np.inf] * len(
                    self.possible_protocols))
        )

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

    def get_current_state_observation(self) -> np.ndarray:
        obs_with_baseline = np.append(
            [self.total_data_to_send, self.amount_sent, self.amount_left, self.data_texture.value],
            self.current_baseline_data.to_numpy().reshape(-1)
        )
        obs_with_sent_per_proto = np.append(
            obs_with_baseline, list(self.sent_values_per_proto.values())
        )
        return obs_with_sent_per_proto

    def reset(self) -> np.ndarray:
        # set the state randomly
        self.total_data_to_send = np.random.choice(self.data_to_send_possible_values)
        self.amount_left = self.total_data_to_send
        self.amount_sent = 0
        self.current_step = 0
        self.data_texture = np.random.choice(DataTextureEnum)
        self.current_baseline_data = self.baseline_datas[np.random.choice(range(len(self.baseline_datas)))]

        self.sent_values_per_proto: OrderedDict[Layer4Protocol, int] = collections.OrderedDict(
            {proto: 0 for proto in self.possible_protocols}
        )

        self.tot_rewards = 0

        # reset the networkIo
        self.network_io.set_baseline_data(self.current_baseline_data)
        self.network_io.reset()

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

        chosen_protocol_idx, chosen_amount_idx = action[0], action[1]

        chosen_protocol_idx = min(chosen_protocol_idx, len(self.possible_protocols) - 1)
        chosen_amount_idx = min(chosen_amount_idx, len(self.legal_packet_sizes) - 1)

        return self.possible_protocols[chosen_protocol_idx], self.legal_packet_sizes[chosen_amount_idx]

    def step_result(self, reward: float, done: bool = False, info=None) -> Tuple[np.ndarray, float, bool, dict]:
        if info is None:
            info = dict()

        self.tot_rewards += reward

        if self.tot_rewards < 0 and self.current_step > 10000:
            info.update(dict(success=False))
            return self.get_current_state_observation(), -self.failure_penalty_size, True, info

        return self.get_current_state_observation(), reward, done, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        chosen_protocol, chosen_amount = self.action_to_proto_amount(action)

        self.current_step += 1

        if not self.is_action_legal(chosen_protocol, chosen_amount):
            return self.step_result(-self.illegal_move_penalty_size)

        if chosen_protocol is None and chosen_amount is None:
            return self.step_result(-self.nop_penalty_size)

        send_res = self.network_io.send(bytes(chosen_amount), chosen_protocol, self.data_texture)

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
