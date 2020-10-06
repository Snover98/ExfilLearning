import click
import numpy as np
import pandas as pd
from pathlib import Path
import itertools

import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
import ray.rllib.agents.a3c.a2c as a2c
import ray.rllib.agents.dqn.apex as apex_dqn
import ray.rllib.agents.dqn.dqn as dqn
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.models import ModelCatalog

from ReinforcementLearning.Models.batch_norm_model import TorchBatchNormModel
from ReinforcementLearning.Models.skip_connection_model import TorchSkipConnectionModel
from ReinforcementLearning.Environments import NonLearningNetworkIoEnv
from NetworkIO import *
from Protocols import *
from ReinforcementLearning.mutators import *

from typing import List, Dict, Any, Iterable, Callable, Set, Optional, Tuple, Union

NetworkIoCreator = Callable[[], BaseNetworkIO]


class RayNonLearningNetworkIoEnv(NonLearningNetworkIoEnv):
    def __init__(self, config):
        super().__init__(**config)


def get_baselines(results_path: Path = None) -> List[pd.DataFrame]:
    if not results_path:
        results_path: Path = Path('G:\\', 'ItzikProject', 'tmp_results', 'sniff_results')

    baseline_datas: List[pd.DataFrame] = [
        pd.read_csv(str(csv_file), index_col=0) for csv_file in results_path.glob('*csv')
    ]

    return baseline_datas


def ensemble_network_io_creator(network_ios_creation_funcs: Iterable[NetworkIoCreator]) -> NetworkIoCreator:
    return lambda: FullConsensusEnsembleNetworkIO(
        [network_io_fn() for network_io_fn in network_ios_creation_funcs]
    )


def env_config(baseline_datas: List[pd.DataFrame], protos_to_drop: Set[str] = None,
               max_deviation_from_protos: Optional[float] = .1, std_coef: Optional[float] = 3.0,
               action_space: str = 'multidiscrete',
               baselines_mutators: List[Mutator] = None, inplace_mutations: Union[bool, List[bool]] = True,
               **other_env_kwargs) -> Dict[str, Any]:
    env_config_dict: Dict[str, Any] = dict()

    if protos_to_drop is None:
        protos_to_drop = {
            f"{proto_name}:{port_number}" for proto_name, port_number in itertools.product(['UDP', 'TCP'], [80, 443])
        }

    # for future mutations of the baseline datas
    if baselines_mutators:
        baseline_datas = mutate_baselines(baseline_datas, baselines_mutators, inplace_mutations)

    env_config_dict['baseline_datas'] = baseline_datas

    # generate the network io creation function
    network_io_fns: List[NetworkIoCreator] = [
        lambda: NotPortProtoNetworkIO(str_to_layer4_proto(proto)) for proto in protos_to_drop
    ]
    network_io_fns.append(AllDataBetweenMinMaxNetworkIO)
    required_ios_indices: List[int] = [len(network_io_fns) - 1]

    if max_deviation_from_protos is not None:
        network_io_fns.append(
            lambda: NoMoreThanXPercentDeviationPerProtoNetworkIO(max_deviation_from_protos=max_deviation_from_protos)
        )

    if std_coef is not None:
        network_io_fns.append(lambda: DataSizeWithinStdOfMeanForProtoNetworkIO(std_coef=std_coef))
        required_ios_indices.append(len(network_io_fns) - 1)

    env_config_dict['network_io_fn'] = ensemble_network_io_creator(network_io_fns)
    env_config_dict['required_io_idx'] = required_ios_indices

    # add action space
    env_config_dict['model_action_space'] = action_space

    # add remaining kwargs
    env_config_dict.update(other_env_kwargs)

    return env_config_dict


def double_middle_drop_lr_sched(lr: float, stop_timesteps: int) -> List[Tuple[int, float]]:
    return [(0, lr), (stop_timesteps // 4, 0.75 * lr), (int(stop_timesteps * .75), 0.75 * lr),
            (int(stop_timesteps * .75) + 1, lr * .125)]


class TrialStopper:
    def __init__(self, stop_timesteps: int = None, stop_iters: int = None, stop_reward: float = None,
                 max_episodes_without_improvement: int = None, change_delta: float = 0.05,
                 timed_thresholds: Sequence[Tuple[int, float]] = None):
        self.stop_timesteps = stop_timesteps
        self.stop_iters = stop_iters
        self.stop_reward = stop_reward
        self.max_episodes_without_improvement = max_episodes_without_improvement
        self.change_delta = change_delta
        self.timed_thresholds = timed_thresholds

        self.cur_episodes_without_improvement = 0
        self.prev_reward = -np.inf
        self.max_reward = -np.inf

    def __call__(self, result):
        self.last_time_step = result['timesteps_total']

        self.max_reward = max(self.max_reward, result['episode_reward_mean'])

        if self.stop_iters and result['training_iteration'] >= self.stop_iters:
            return True

        if self.stop_timesteps and result['timesteps_total'] >= self.stop_timesteps:
            return True

        if self.stop_reward and result['episode_reward_mean'] >= self.stop_reward:
            return True

        if self.max_episodes_without_improvement:
            if result['episode_reward_mean'] - self.prev_reward <= self.change_delta:
                self.cur_episodes_without_improvement += 1
            else:
                self.cur_episodes_without_improvement = 0

            self.prev_reward = result['episode_reward_mean']

            if self.cur_episodes_without_improvement >= self.max_episodes_without_improvement:
                return True

        if self.timed_thresholds:
            relevant_rewards = [reward_thresh for time_step, reward_thresh in self.timed_thresholds if
                                result['timesteps_total'] >= time_step] + [-np.inf]
            if self.max_reward < max(relevant_rewards):
                return True

        return False


class Stopper:
    def __init__(self, stop_timesteps: int = None, stop_iters: int = None, stop_reward: float = None,
                 max_episodes_without_improvement: int = None, change_delta: float = 0.05,
                 timed_thresholds: Sequence[Tuple[int, float]] = None):
        self.stop_timesteps = stop_timesteps
        self.stop_iters = stop_iters
        self.stop_reward = stop_reward
        self.max_episodes_without_improvement = max_episodes_without_improvement
        self.change_delta = change_delta
        self.timed_thresholds = timed_thresholds
        self.stoppers = dict()

    def __call__(self, trial_id, result):
        if trial_id not in self.stoppers:
            self.stoppers[trial_id] = TrialStopper(self.stop_timesteps, self.stop_iters, self.stop_reward,
                                                   self.max_episodes_without_improvement, self.change_delta,
                                                   self.timed_thresholds)

        stop_trial = self.stoppers[trial_id](result)

        if stop_trial:
            del self.stoppers[trial_id]

        return stop_trial


@click.command("Run training with the non-learning environment")
@click.option('-a', '--algo', 'algorithm', type=click.STRING, help='The algorithm to run', default='A2C')
@click.option('-i', "--stop-iters", 'stop_iters', type=click.INT, default=100)
@click.option('-t', "--stop-timesteps", 'stop_timesteps', type=click.INT, default=200000)
@click.option("--stop-reward", 'stop_reward', type=click.FloatRange(-np.inf, 1.0), default=1.0)
@click.option('--as-test', 'as_test', is_flag=True)
@click.option('--baselines-path', 'baselines_path', type=click.Path(exists=True, file_okay=False, resolve_path=True),
              default=None)
@click.option('-n', '--num-samples', 'num_samples', type=click.INT, default=1)
@click.option('-s', '--seed', 'seed', type=click.INT, default=None)
def main(algorithm: str, stop_iters: int, stop_timesteps: int, stop_reward: float, as_test: bool, baselines_path: str,
         num_samples: int,
         seed: int) -> None:
    algorithm = algorithm.upper()

    ModelCatalog.register_custom_model("bn_model", TorchBatchNormModel)
    ModelCatalog.register_custom_model("skip_model", TorchSkipConnectionModel)

    ray.init(include_dashboard=False)

    if baselines_path:
        baselines_path = Path(baselines_path)
    baseline_datas: List[pd.DataFrame] = get_baselines(baselines_path)

    action_space: str = 'multidiscrete'

    timed_thresholds: Optional[Sequence[Tuple[int, float]]] = None

    config: Dict[str, Any] = dict()
    if algorithm == 'A2C':
        config.update(a2c.A2C_DEFAULT_CONFIG)

        config["rollout_fragment_length"] = tune.grid_search([20, 50, 100])
        config["use_gae"] = False
        config['vf_loss_coeff'] = .25
        config["lr"] = 0.01
        config['model']['fcnet_hiddens'] = [1024, 512, 256]
        config['min_iter_time_s'] = 20
        # timed_thresholds = [(int(1e4), -1), (int(5e4), 0), (int(1e5), .5)]
        timed_thresholds = [(int(1e4), -4.5), (int(2e4), -2), (int(6e4), 0), (int(1e5), .5), (int(1.5e5), .7),
                            (int(2.5e5), .9)]
    elif algorithm == 'APEX':
        config.update(apex_dqn.APEX_DEFAULT_CONFIG)
        action_space = 'discrete'
    elif algorithm == 'PPO':
        config.update(ppo.DEFAULT_CONFIG)
        config["lr"] = 1e-5
        config['entropy_coeff'] = 0.01
        config['clip_param'] = .3
        config['model']['fcnet_hiddens'] = [1024, 512, 256]
        config["rollout_fragment_length"] = 10
    elif algorithm == 'DQN':
        config.update(dqn.DEFAULT_CONFIG)

        config['hiddens'] = tune.grid_search([[256], [256, 256]])
        config['grad_clip'] = tune.grid_search([.5, 40])
        action_space = 'discrete'
    elif algorithm == 'RAINBOW':
        algorithm = 'DQN'
        config['n_step'] = tune.grid_search([2, 5, 10])
        config['noisy'] = True
        config['num_atoms'] = tune.grid_search([2, 5, 10])
        config['v_min'] = -5.0
        config['v_max'] = 1.0
        # config["sigma0"] = tune.grid_search([])

        config['hiddens'] = tune.grid_search([[256], [256, 256]])
        config['grad_clip'] = tune.grid_search([.5, 40])
        action_space = 'discrete'

    config.update({
        "env": RayNonLearningNetworkIoEnv,
        "env_config": env_config(
            baseline_datas, action_space=action_space,
            std_coef=None,
            use_random_io_mask=True,
            baselines_mutators=[
                sizes_mult_mutator(8),
                # switch_2_protocols_mutation,
                # shuffle_protocols_mutation
            ],
            # inplace_mutations=[True, False],
        ),
        # "framework": "torch",
        "num_gpus": 0,
        "num_envs_per_worker": 4,
        'num_workers': 1
    })

    lrs = sorted(c * 10 ** -i for i, c in itertools.product(range(2, 6), [1, 5]))
    # config["lr"] = tune.grid_search(lrs)
    # config['lr_schedule'] = tune.grid_search(
    #     [double_middle_drop_lr_sched(lr, stop_timesteps) for lr in lrs] + [[(0, lr)] for lr in lrs]
    # )

    config['lr_schedule'] = double_middle_drop_lr_sched(config["lr"], stop_timesteps)
    # config['lr_schedule'] = tune.grid_search([None, config['lr_schedule']])

    rand_seeds = [238749400, 1550590419, 306522394, 664536080, 827704252, 1927293810, 1015498960, 285322805, 552328904,
                  1913151724, 841076802, 1554963668, 793707278, 692496376, 169558613, 931430758, 653645527, 115908151,
                  643336564, 262074737]
    config['seed'] = tune.grid_search(rand_seeds)
    # config['seed'] = seed

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        # "episode_reward_mean": stop_reward,
    }

    stop = Stopper(
        stop_timesteps=stop_timesteps,
        stop_iters=stop_iters,
        # stop_reward=stop_reward,
        max_episodes_without_improvement=30,
        timed_thresholds=timed_thresholds
    )

    results = tune.run(algorithm, config=config, stop=stop, num_samples=num_samples)

    if as_test:
        check_learning_achieved(results, stop_reward)

    ray.shutdown()


if __name__ == '__main__':
    main()
