import os
import pickle
import pandas as pd
import numpy as np
import itertools
import tensorflow as tf
import warnings
from tqdm import tqdm
import sys

from ctypes import windll

from stable_baselines import A2C, PPO2, ACKTR, HER, SAC
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines.common.callbacks import EvalCallback, BaseCallback
from stable_baselines.common import BaseRLModel
from sklearn.model_selection import train_test_split, KFold

from ReinforcementLearning.Environments import *
from NetworkIO import *
from Protocols import *

from typing import Set, List, Tuple, Optional, Callable

DEFAULT_RAND_STATE_PATH = 'rand_state.pickle'

MULTI_DISCRETE_MODELS = (A2C, PPO2)


class PbarCallback(BaseCallback):
    def __init__(self, pbar: tqdm, verbose=0):
        super().__init__(verbose)
        self.pbar = pbar

    def _on_step(self) -> bool:
        self.pbar.update()
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


class EvalPbarCallback:
    def __init__(self, pbar: tqdm):
        self.num_successes: int = 0
        self.pbar: tqdm = pbar

    def __call__(self, locs, _) -> None:
        if locs['done'].item():
            self.pbar.update()
            self.num_successes += sum(int(info.get('success', False)) for info in locs['_info'])


class Trainer:
    def __init__(self, model_fn: Callable[[VecEnv], BaseRLModel], train_baseline_datas: List[pd.DataFrame],
                 eval_baseline_datas: List[pd.DataFrame], model_name: str = None, total_train_steps: int = int(1e5),
                 eval_episodes: int = 50, model_uses_box: bool = False, use_eval_callback: bool = False,
                 pop_up_window: bool = False, run_number: int = None, rand_state_path: str = DEFAULT_RAND_STATE_PATH):
        self.model_fn: Callable[[VecEnv], BaseRLModel] = model_fn
        self.train_baseline_datas: List[pd.DataFrame] = train_baseline_datas
        self.eval_baseline_datas: List[pd.DataFrame] = eval_baseline_datas
        self._model_name: Optional[str] = model_name
        self.total_train_steps: int = total_train_steps
        self.eval_episodes: int = eval_episodes
        self.model_uses_box: bool = model_uses_box
        self.use_eval_callback: bool = use_eval_callback
        self.pop_up_window: bool = pop_up_window
        self.run_number: Optional[int] = run_number
        self.rand_state_path: str = rand_state_path

        self.train_env: DummyVecEnv
        self.eval_env: DummyVecEnv
        self.train_env, self.eval_env = self.create_train_eval_envs(use_box=self.model_uses_box)

        self._model: Optional[BaseRLModel] = None

    @staticmethod
    def create_env(baseline_datas: List[pd.DataFrame], protos_to_drop: Set[str], **env_kwargs) -> DummyVecEnv:
        return DummyVecEnv([
            lambda: NonLearningNetworkIoEnv(
                baseline_datas,
                network_io_fn=lambda: FullConsensusEnsembleNetworkIO(
                    [
                        *(NotPortProtoNetworkIO(str_to_layer4_proto(proto)) for proto in protos_to_drop),
                        NoMoreThanXPercentDeviationPerProtoNetworkIO()
                    ]
                ),
                **env_kwargs
            )
        ])

    def create_train_eval_envs(self, **env_kwargs) -> Tuple[DummyVecEnv, DummyVecEnv]:
        protos_to_drop: Set[str] = {
            "UDP:443",
            "TCP:443",
            "UDP:80",
            "TCP:80"
        }

        baseline_datas = self.train_baseline_datas + self.eval_baseline_datas

        max_pow_of_2_to_send: int = min(
            np.log2(baseline_data.drop(protos_to_drop.intersection(baseline_data.index)).total_bytes.sum() * .1).astype(
                np.int) for baseline_data in baseline_datas
        )
        valid_send_amounts: List[int] = [2 ** i for i in range(10, max_pow_of_2_to_send + 1)]

        all_protos: Set[str] = set(
            itertools.chain.from_iterable(baseline_data.index for baseline_data in baseline_datas))

        env: DummyVecEnv = self.create_env(self.train_baseline_datas, protos_to_drop, all_protos=all_protos,
                                           data_to_send_possible_values=valid_send_amounts, **env_kwargs)

        eval_env: DummyVecEnv = self.create_env(self.eval_baseline_datas, protos_to_drop, all_protos=all_protos,
                                                data_to_send_possible_values=valid_send_amounts, **env_kwargs)

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
            if hasattr(model, 'lr_schedule'):
                model_name_parts += ("lr_sched", model.lr_schedule)
            if self.run_number is not None:
                model_name_parts += ("run", str(self.run_number))
            self._model_name = '_'.join(model_name_parts)

        return self._model_name

    def create_callbacks(self, eval_env: DummyVecEnv) -> List[BaseCallback]:
        callbacks: List[BaseCallback] = list()

        if self.use_eval_callback:
            model_path: str = os.path.join('.', 'non_learning_io_logs', self.model_name, "")
            eval_callback = EvalCallback(eval_env, best_model_save_path=model_path, log_path=model_path,
                                         eval_freq=10000, verbose=0, n_eval_episodes=20,
                                         deterministic=True, render=False)
            callbacks.append(eval_callback)

        callbacks.append(
            PbarCallback(tqdm(desc="Training Steps Progress", total=self.total_train_steps, file=sys.stdout)))

        return callbacks

    def run_train(self, load_rand_state: bool = False) -> None:
        if load_rand_state:
            load_numpy_rand_state(self.rand_state_path)

        print("=" * 60)
        print(f"MODEL NAME:\t{self.model_name}")
        print("=" * 60)

        callbacks: List[BaseCallback] = self.create_callbacks(self.eval_env)

        self.model.learn(self.total_train_steps, tb_log_name=self.model_name, callback=callbacks)

        if self.use_eval_callback:
            self.model = type(self.model).load(f'.\\non_learning_io_logs\\{self.model_name}\\best_model.zip')

        eval_pbar = tqdm(desc="Evaluation Episodes Progress", total=self.eval_episodes, file=sys.stdout)

        eval_callback = EvalPbarCallback(eval_pbar)

        mean_reward: float
        std_reward: float
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, self.eval_episodes,
            callback=eval_callback,
        )
        eval_pbar.close()
        print(f"mean reward:\t\t\t{mean_reward:.2f}")
        print(f"reward std:\t\t\t\t{std_reward:.2f}")

        num_successes: int = eval_callback.num_successes
        success_percentage: float = num_successes / self.eval_episodes * 100

        successes_msg: str = f"number of successes:\t{num_successes}/{self.eval_episodes} ({success_percentage : .2f}%)"
        print(successes_msg)

        if self.pop_up_window:
            windll.user32.MessageBoxW(0, successes_msg, "Finished Evaluation!", 1)


def train_model(model_fn, baseline_datas: List[pd.DataFrame], model_name: str = None,
                total_train_steps: int = int(1e5), eval_episodes: int = 100, model_uses_box: bool = None,
                use_eval_callback: bool = False, pop_up_window: bool = False, eval_amount: float = None,
                n_splits: int = None, state_pickle_path: str = DEFAULT_RAND_STATE_PATH):
    train_baseline_datas: List[pd.DataFrame]
    eval_baseline_datas: List[pd.DataFrame]

    if n_splits is not None:
        for run_number, (train_indices, eval_indices) in enumerate(KFold(n_splits).split(baseline_datas), 1):
            train_baseline_datas = [baseline_datas[i] for i in train_indices]
            eval_baseline_datas = [baseline_datas[i] for i in eval_indices]

            if model_name is not None:
                cur_model_name = f"{model_name}_run_{run_number}"
            else:
                cur_model_name = None

            trainer = Trainer(model_fn, train_baseline_datas, eval_baseline_datas, model_name=cur_model_name,
                              total_train_steps=total_train_steps, eval_episodes=eval_episodes,
                              model_uses_box=model_uses_box, use_eval_callback=use_eval_callback,
                              pop_up_window=pop_up_window, run_number=run_number, rand_state_path=state_pickle_path)
            trainer.run_train(load_rand_state=True)
        return

    if eval_amount is None:
        train_baseline_datas = baseline_datas
        eval_baseline_datas = baseline_datas
    else:
        train_baseline_datas, eval_baseline_datas = train_test_split(baseline_datas, test_size=eval_amount)

    trainer = Trainer(model_fn, train_baseline_datas, eval_baseline_datas, model_name=model_name,
                      total_train_steps=total_train_steps, eval_episodes=eval_episodes,
                      model_uses_box=model_uses_box, use_eval_callback=use_eval_callback, pop_up_window=pop_up_window,
                      rand_state_path=state_pickle_path)
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


def main(state_pickle_path: str = DEFAULT_RAND_STATE_PATH, run_name: str = None) -> None:
    load_numpy_rand_state(state_pickle_path)

    seed: int = np.random.randint(np.iinfo(np.int).max)
    tot_train_steps = int(1e5)
    print(f"The seed is {seed}")

    results_path: str = os.path.join('G:\\', 'ItzikProject', 'tmp_results', 'sniff_results')
    csv_file_names: List[str] = [f'bucket_{i}_results.csv' for i in range(1, 25)]

    baseline_datas: List[pd.DataFrame] = [
        pd.read_csv(os.path.join(results_path, csv_file_name), index_col=0) for csv_file_name in csv_file_names
    ]

    # baseline_data = baseline_data.drop(baseline_data[baseline_data['packet_size_std_bytes'] == 0].index)

    legal_policies = ('MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy')
    # lr_schedules = ('linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop')
    lr_schedules = ('double_middle_drop',)

    train_model(
        lambda env: create_model_fn(A2C, 'MlpLnLstmPolicy', env, seed, lr_schedule='double_middle_drop'),
        baseline_datas,
        n_splits=5,
        total_train_steps=tot_train_steps,
        use_eval_callback=True,
        model_name=run_name
    )

    # for legal_policy, lr_sched in itertools.product(legal_policies, lr_schedules):
    #     train_model(lambda env: create_model_fn(A2C, legal_policy, env, seed, lr_schedule=lr_sched),
    #                 baseline_datas,
    #                 seed=seed,
    #                 eval_amount=.25,
    #                 use_eval_callback=False,
    #                 total_train_steps=tot_train_steps)

    # windll.user32.MessageBoxW(0,
    #                           "Finished all models!",
    #                           "Finished Evaluation!", 0x1000)


if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    for idx in range(1, 6):
        print("=" * 70)
        print(f"SEED NUMBER:\t{idx}")
        main(f'rand_state_{idx}.pickle', run_name=f"seed_{idx}")
