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
import ray.rllib.agents.ppo.ppo as ppo

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

    if max_deviation_from_protos is not None:
        network_io_fns.append(
            lambda: NoMoreThanXPercentDeviationPerProtoNetworkIO(max_deviation_from_protos=max_deviation_from_protos)
        )

    if std_coef is not None:
        network_io_fns.append(lambda: DataSizeWithinStdOfMeanForProtoNetworkIO(std_coef=std_coef))

    env_config_dict['network_io_fn'] = ensemble_network_io_creator(network_io_fns)

    # add action space
    env_config_dict['model_action_space'] = action_space

    # add remaining kwargs
    env_config_dict.update(other_env_kwargs)

    return env_config_dict


def double_middle_drop_lr_sched(lr: float, stop_timesteps: int) -> List[Tuple[int, float]]:
    return [(0, lr), (stop_timesteps // 4, 0.75 * lr), (int(stop_timesteps * .75), 0.75 * lr),
            (int(stop_timesteps * .75) + 1, lr * .125)]


@click.command("Run training with the non-learning environment")
@click.option('--algo', '-a', 'algorithm', type=click.STRING, help='The algorithm to run', default='A2C')
@click.option("--stop-iters", 'stop_iters', type=click.INT, default=100)
@click.option("--stop-timesteps", 'stop_timesteps', type=click.INT, default=200000)
@click.option("--stop-reward", 'stop_reward', type=click.FloatRange(-np.inf, 1.0), default=1.0)
@click.option('--as-test', 'as_test', is_flag=True)
@click.option('--baselines-path', 'baselines_path', type=click.Path(exists=True, file_okay=False, resolve_path=True),
              default=None)
@click.option('-s', '--seed', 'seed', type=click.INT, default=None)
def main(algorithm: str, stop_iters: int, stop_timesteps: int, stop_reward: float, as_test: bool, baselines_path: str,
         seed: int) -> None:
    ray.init(include_dashboard=False)

    if baselines_path:
        baselines_path = Path(baselines_path)
    baseline_datas: List[pd.DataFrame] = get_baselines(baselines_path)

    action_space: str = 'multidiscrete'

    config: Dict[str, Any] = dict()
    if algorithm == 'A2C':
        config.update(a2c.A2C_DEFAULT_CONFIG)
        # config["use_gae"] = tune.grid_search([False, True])
        # config["model"]['fcnet_hiddens'] = tune.grid_search([[256, 256], [256, 256, 256]])
        # config['model']['use_lstm'] = tune.grid_search([False, True])
        # config['vf_loss_coeff'] = tune.grid_search([.25, config['vf_loss_coeff']])
        # config['grad_clip'] = tune.grid_search([.5, config['grad_clip']])
        # config["rollout_fragment_length"] = tune.grid_search([5, config["rollout_fragment_length"]])

        config["rollout_fragment_length"] = tune.grid_search([5, 10, 20])
        config['model']['use_lstm'] = False
        config["use_gae"] = False
        config['vf_loss_coeff'] = .25
        # config['grad_clip'] = .5
    elif algorithm == 'APEX':
        config.update(apex_dqn.APEX_DEFAULT_CONFIG)
        config['model']['use_lstm'] = tune.grid_search([False, True])
        config['num_workers'] = 2
        action_space = 'discrete'
    elif algorithm == 'PPO':
        config.update(ppo.DEFAULT_CONFIG)

    config.update({
        "env": RayNonLearningNetworkIoEnv,
        "env_config": env_config(baseline_datas, action_space=action_space,
                                 std_coef=None,
                                 use_random_io_mask=False,
                                 # baselines_mutators=[
                                 #     sizes_mult_mutator(8),
                                 # ]
                                 ),
        "framework": "torch",
        "num_gpus": 0
    })

    lrs = sorted(c * 10 ** -i for i, c in itertools.product(range(2, 6), [1, 5]))
    # config["lr"] = tune.grid_search(lrs)
    config['lr_schedule'] = tune.grid_search(
        [double_middle_drop_lr_sched(lr, stop_timesteps) for lr in lrs] + [[(0, lr)] for lr in lrs]
    )

    # config["lr"] = 0.01
    # config['lr_schedule'] = double_middle_drop_lr_sched(config["lr"], stop_timesteps)
    # config['lr_schedule'] = tune.grid_search([None, config['lr_schedule']])

    rand_seeds = [238749400, 1550590419, 306522394, 664536080, 827704252, 1927293810, 1015498960, 285322805, 552328904,
                  1913151724, 841076802, 1554963668, 793707278, 692496376, 169558613, 931430758, 653645527, 115908151,
                  643336564, 262074737]
    # config['seed'] = tune.grid_search(rand_seeds)
    config['seed'] = seed

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        # "episode_reward_mean": stop_reward,
    }

    results = tune.run(algorithm, config=config, stop=stop, reuse_actors=True)

    if as_test:
        check_learning_achieved(results, stop_reward)

    ray.shutdown()


if __name__ == '__main__':
    main()
