import os
import sys
import itertools
import pickle
import sys
import warnings
from copy import deepcopy
from enum import Enum
from typing import Set, List, Tuple, Optional, Dict, Any, Type, Iterable, NamedTuple

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from stable_baselines import *
from stable_baselines.common import BaseRLModel
from stable_baselines.common.callbacks import EvalCallback, BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from tqdm import tqdm

from NetworkIO import *
from Protocols import *
from ReinforcementLearning.Environments import NonLearningNetworkIoEnv

RAND_STATES_DEFAULT_DIR = 'rand_states'
DEFAULT_RAND_STATE_PATH = os.path.join(RAND_STATES_DEFAULT_DIR, 'rand_state.pickle')


class ChosenAlgoEnum(Enum):
    PPO = 'ppo'
    A2C = 'a2c'
    ACKTR = 'acktr'
    SAC = 'sac'
    DQN = 'dqn'


MULTI_DISCRETE_MODELS: Set[Type[BaseRLModel]] = {A2C, PPO1, PPO2, TRPO, ACKTR}
BOX_MODELS: Set[Type[BaseRLModel]] = {DDPG, SAC, TD3}


class ModelCreator:
    def __init__(self, model_cls: Type[BaseRLModel], policy: str, seed: int, *args, **kwargs):
        self.model_cls: Type[BaseRLModel] = model_cls
        self.policy: str = policy
        self.seed: int = seed
        self.args: Iterable[Any] = args
        self.kwargs: Dict[str, Any] = kwargs

    @property
    def model_action_space(self) -> str:
        if self.model_cls in MULTI_DISCRETE_MODELS:
            return 'multidiscrete'
        elif self.model_cls in BOX_MODELS:
            return 'box'
        else:
            return 'discrete'

    def __call__(self, env: VecEnv, *args, **kwargs) -> BaseRLModel:
        tot_kwargs = deepcopy(self.kwargs)
        tot_kwargs.update(kwargs)

        tot_args = itertools.chain(self.args, args)

        return self.model_cls(self.policy, env, *tot_args, seed=self.seed, **tot_kwargs)


class PbarCallback(BaseCallback):
    def __init__(self, pbar: tqdm, num_envs: int = 1, verbose=0):
        super().__init__(verbose)
        self.pbar: tqdm = pbar
        self.num_envs: int = num_envs

    def _on_step(self) -> bool:
        self.pbar.update(self.num_envs)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


class EvalPbarCallback:
    def __init__(self, pbar: tqdm, verbose: bool = True):
        self.num_successes: int = 0
        self.pbar: tqdm = pbar
        self.verbose: bool = verbose

    def __call__(self, locs, _) -> None:
        if locs['done'].item():
            if self.verbose:
                self.pbar.update()

            self.num_successes += sum(int(info.get('success', False)) for info in locs['_info'])


class TrainResult(NamedTuple):
    mean_reward: float
    std_reward: float
    num_successes: int
    num_episodes: int

    def res_msg(self) -> str:
        success_prt = self.num_successes / self.num_episodes
        return '\n'.join(
            (
                f"mean reward:\t\t\t{self.mean_reward:.2f}",
                f"reward std:\t\t\t\t{self.std_reward:.2f}",
                f"number of successes:\t{self.num_successes}/{self.num_episodes} ({success_prt:.2%})"
            )
        )


class Trainer:
    def __init__(self, model_creator: ModelCreator, train_baseline_datas: List[pd.DataFrame],
                 eval_baseline_datas: List[pd.DataFrame], model_name: str = None, total_train_steps: int = int(1e5),
                 eval_episodes: int = 50, use_eval_callback: bool = False, run_number: int = None,
                 rand_state_path: str = DEFAULT_RAND_STATE_PATH, n_envs: int = 1, verbose: bool = False):
        self.model_fn: ModelCreator = model_creator
        self.train_baseline_datas: List[pd.DataFrame] = train_baseline_datas
        self.eval_baseline_datas: List[pd.DataFrame] = eval_baseline_datas
        self._model_name: Optional[str] = model_name
        self.total_train_steps: int = total_train_steps
        self.eval_episodes: int = eval_episodes
        self.use_eval_callback: bool = use_eval_callback
        self.run_number: Optional[int] = run_number
        self.rand_state_path: str = rand_state_path
        self.n_envs: int = n_envs
        self.verbose: bool = verbose

        self.train_env: VecEnv
        self.eval_env: VecEnv
        self.train_env, self.eval_env = self.create_train_eval_envs(model_action_space=model_creator.model_action_space)

        self._model: Optional[BaseRLModel] = None

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @staticmethod
    def create_env(baseline_datas: List[pd.DataFrame], protos_to_drop: Set[str],
                   n_envs: int = 1, **env_kwargs) -> DummyVecEnv:
        return DummyVecEnv([
            lambda: NonLearningNetworkIoEnv(
                baseline_datas,
                network_io_fn=lambda: FullConsensusEnsembleNetworkIO(
                    [
                        *(NotPortProtoNetworkIO(str_to_layer4_proto(proto)) for proto in protos_to_drop),
                        NoMoreThanXPercentDeviationPerProtoNetworkIO(max_deviation_from_protos=.10),
                        # DataSizeWithinStdOfMeanForProtoNetworkIO(std_coef=3.0)
                    ]
                ),
                **env_kwargs
            ) for _ in range(n_envs)
        ])

    def create_train_eval_envs(self, **env_kwargs) -> Tuple[VecEnv, VecEnv]:
        protos_to_drop: Set[str] = {
            f"{proto_name}:{port_number}" for proto_name, port_number in itertools.product(['UDP', 'TCP'], [80, 443])
        }

        baseline_datas = self.train_baseline_datas + self.eval_baseline_datas

        # no_std_protos: Set[str] = set(
        #     # itertools.chain.from_iterable(
        #     #     (
        #     #         proto for proto in baseline_data.index if baseline_data.loc[str(proto)].packet_size_std_bytes == 0.0
        #     #     ) for baseline_data in baseline_datas
        #     # )
        # )
        #
        # max_pow_of_2_to_send: int = min(
        #     np.log2(baseline_data.drop(
        #         (protos_to_drop | no_std_protos).intersection(baseline_data.index)).total_bytes.sum() * .10).astype(
        #         np.int) for baseline_data in baseline_datas
        # )

        # valid_send_amounts: List[int] = [2 ** i for i in range(10, max_pow_of_2_to_send + 1)]

        all_protos: Set[str] = set(
            itertools.chain.from_iterable(baseline_data.index for baseline_data in baseline_datas))

        env: VecEnv = self.create_env(self.train_baseline_datas, protos_to_drop, all_protos=all_protos,
                                      # data_to_send_possible_values=valid_send_amounts,
                                      **env_kwargs, n_envs=self.n_envs)

        eval_env: VecEnv = self.create_env(self.eval_baseline_datas, protos_to_drop, all_protos=all_protos,
                                           # data_to_send_possible_values=valid_send_amounts,
                                           **env_kwargs)

        return env, eval_env

    @property
    def model(self) -> BaseRLModel:
        if self._model is None:
            self._model = self.model_fn(self.train_env)

        return self._model

    @model.setter
    def model(self, new_model: BaseRLModel):
        self._model = new_model

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            model = self.model
            model_name_parts: Tuple[str, ...] = (type(model).__name__, model.policy.__name__)
            # if hasattr(model, 'lr_schedule'):
            #     model_name_parts += ("lr_sched", model.lr_schedule)
            if self.run_number is not None:
                model_name_parts += ("run", str(self.run_number))
            self._model_name = '_'.join(model_name_parts)

        return self._model_name

    def create_callbacks(self, eval_env: VecEnv) -> List[BaseCallback]:
        callbacks: List[BaseCallback] = list()

        if self.use_eval_callback:
            model_path: str = os.path.join('non_learning_io_logs', self.model_name, "")
            eval_callback = EvalCallback(eval_env, best_model_save_path=model_path, log_path=model_path,
                                         eval_freq=2 ** 13, verbose=0, n_eval_episodes=32,
                                         deterministic=True, render=False)
            callbacks.append(eval_callback)

        if self.verbose:
            callbacks.append(
                PbarCallback(
                    tqdm(desc="Training Steps Progress",
                         total=self.total_train_steps,
                         file=sys.stdout),
                    num_envs=self.n_envs
                )
            )

        return callbacks

    def run_train(self, load_rand_state: bool = False) -> TrainResult:
        if load_rand_state:
            load_numpy_rand_state(self.rand_state_path)

        self.print("=" * 60)
        self.print(f"MODEL NAME:\t{self.model_name}")
        self.print("=" * 60)

        callbacks: List[BaseCallback] = self.create_callbacks(self.eval_env)

        self.model.learn(self.total_train_steps, tb_log_name=self.model_name, callback=callbacks)

        if self.use_eval_callback:
            self.model = type(self.model).load(os.path.join('non_learning_io_logs', self.model_name, "best_model.zip"))

        eval_pbar = tqdm(desc="Evaluation Episodes Progress", total=self.eval_episodes, file=sys.stdout)

        eval_callback = EvalPbarCallback(eval_pbar, self.verbose)

        mean_reward: float
        std_reward: float
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, self.eval_episodes,
            callback=eval_callback,
        )
        eval_pbar.close()

        result = TrainResult(mean_reward, std_reward, eval_callback.num_successes, self.eval_episodes)
        self.print(result.res_msg())

        return result


def train_model(model_creator: ModelCreator, baseline_datas: List[pd.DataFrame], model_name: str = None,
                total_train_steps: int = int(2 ** 17), eval_episodes: int = 1024, use_eval_callback: bool = False,
                eval_amount: float = None, n_splits: int = None, state_pickle_path: str = DEFAULT_RAND_STATE_PATH,
                n_envs: int = 1, verbose: bool = False):
    train_baseline_datas: List[pd.DataFrame]
    eval_baseline_datas: List[pd.DataFrame]

    if n_splits is not None:
        train_results: List[TrainResult] = list()

        for run_number, (train_indices, eval_indices) in enumerate(KFold(n_splits).split(baseline_datas), 1):
            train_baseline_datas = [baseline_datas[i] for i in train_indices]
            eval_baseline_datas = [baseline_datas[i] for i in eval_indices]

            if model_name is not None:
                cur_model_name = f"{model_name}_run_{run_number}"
            else:
                cur_model_name = None

            trainer = Trainer(model_creator, train_baseline_datas, eval_baseline_datas, model_name=cur_model_name,
                              total_train_steps=total_train_steps, eval_episodes=eval_episodes,
                              use_eval_callback=use_eval_callback, run_number=run_number,
                              rand_state_path=state_pickle_path, n_envs=n_envs, verbose=verbose)
            result = trainer.run_train(load_rand_state=True)
            train_results.append(result)

        return train_results

    if eval_amount is None:
        train_baseline_datas = baseline_datas
        eval_baseline_datas = baseline_datas
    else:
        train_baseline_datas, eval_baseline_datas = train_test_split(baseline_datas, test_size=eval_amount)

    trainer = Trainer(model_creator, train_baseline_datas, eval_baseline_datas, model_name=model_name,
                      total_train_steps=total_train_steps, eval_episodes=eval_episodes,
                      use_eval_callback=use_eval_callback, rand_state_path=state_pickle_path, n_envs=n_envs,
                      verbose=verbose)
    trainer.run_train(load_rand_state=True)


def create_model_fn(model_cls, policy: str, env: DummyVecEnv, seed: int, **kwargs) -> BaseRLModel:
    return model_cls(policy, env, seed=seed, tensorboard_log=r".\non_learning_io_tensorboard\\", **kwargs)


def save_numpy_rand_state(state_pickle_path: str = DEFAULT_RAND_STATE_PATH) -> None:
    with open(state_pickle_path, 'wb') as state_file:
        pickle.dump(np.random.get_state(), state_file)


def get_numpy_rand_state(state_pickle_path: str):
    with open(state_pickle_path, 'rb') as state_file:
        return pickle.load(state_file)


def load_numpy_rand_state(state_pickle_path: str = DEFAULT_RAND_STATE_PATH) -> None:
    np.random.set_state(get_numpy_rand_state(state_pickle_path))


def main(state_pickle_path: str = DEFAULT_RAND_STATE_PATH, run_name: str = None,
         chosen_algo: ChosenAlgoEnum = ChosenAlgoEnum.A2C) -> None:
    if not os.path.exists(state_pickle_path):
        save_numpy_rand_state(state_pickle_path)

    load_numpy_rand_state(state_pickle_path)

    seed: int = np.random.randint(np.iinfo(np.int).max)
    n_envs: int = 1
    tot_train_steps = int(2 ** 17) * n_envs
    print(f"The seed is {seed}")

    results_path: str = os.path.join('G:\\', 'ItzikProject', 'tmp_results', 'sniff_results')
    csv_file_names: List[str] = [f'bucket_{i}_results.csv' for i in range(1, 25)]

    baseline_datas: List[pd.DataFrame] = [
        pd.read_csv(os.path.join(results_path, csv_file_name), index_col=0) for csv_file_name in csv_file_names
    ]

    mult_factor = 2 ** 3
    for baseline_data in baseline_datas:
        baseline_data[['total_bytes', 'num_packets']] *= mult_factor

    # baseline_data = baseline_data.drop(baseline_data[baseline_data['packet_size_std_bytes'] == 0].index)

    legal_actor_critic_policies = ('MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy')
    # lr_schedules = ('linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop')
    lr_schedules = ('double_middle_drop',)

    ppo_creator = ModelCreator(PPO2, 'MlpLnLstmPolicy', seed, tensorboard_log=r".\non_learning_io_tensorboard\\",
                               nminibatches=min(n_envs, 4))
    a2c_creator = ModelCreator(A2C, 'MlpLnLstmPolicy', seed, tensorboard_log=r".\non_learning_io_tensorboard\\",
                               lr_schedule='double_middle_drop', learning_rate=7e-3)
    acktr_creator = ModelCreator(ACKTR, 'MlpLstmPolicy', seed, tensorboard_log=r".\non_learning_io_tensorboard\\",
                                 lr_schedule='double_middle_drop')
    sac_creator = ModelCreator(SAC, 'MlpPolicy', seed, tensorboard_log=r".\non_learning_io_tensorboard\\",
                               random_exploration=0.0)
    dqn_creator = ModelCreator(DQN, 'MlpPolicy', seed, tensorboard_log=r".\non_learning_io_tensorboard\\")

    creator: ModelCreator
    if not chosen_algo:
        creator = a2c_creator
    elif chosen_algo is chosen_algo.PPO:
        creator = ppo_creator
    elif chosen_algo is chosen_algo.A2C:
        creator = a2c_creator
    elif chosen_algo is chosen_algo.ACKTR:
        creator = acktr_creator
    elif chosen_algo is chosen_algo.SAC:
        creator = sac_creator
    elif chosen_algo is chosen_algo.DQN:
        creator = dqn_creator
    else:
        print("NO CHOSEN ALGO!!!!")
        return

    train_model(
        creator,
        baseline_datas,
        # eval_amount=.25,
        n_splits=5,
        total_train_steps=tot_train_steps,
        use_eval_callback=True,
        model_name=run_name,
        n_envs=n_envs,
        verbose=True
    )


@click.command()
@click.argument('algo', type=click.Choice([algo_e.name for algo_e in ChosenAlgoEnum], case_sensitive=False))
def command(algo: str):
    try:
        main(chosen_algo=ChosenAlgoEnum[algo.upper()])
    except KeyError:
        print(f"No such algorithm {algo}")


if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # save_numpy_rand_state()
    command()

    # for idx in range(1, 6):
    #     print("=" * 70)
    #     print(f"SEED NUMBER:\t{idx}")
    #     main(os.path.join(RAND_STATES_DEFAULT_DIR, f'rand_state_{idx}.pickle'), run_name=f"seed_{idx}")
