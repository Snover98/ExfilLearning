import pandas as pd
from os import urandom, path
import itertools
from multiprocessing import Pool
import tqdm

from Protocols import Layer4Protocol
from ExfilData import ExfilData, DataTextureEnum
from ExfilPlanner import BaseExfilPlanner
from NetworkIO import BaseNetworkIO
from Factories import *

from ExfilPlanner import NaiveMaxDataProtocolExfilPlanner, NaiveXPercentExfilPlanner, \
    NaiveProportionalWeightsRandomExfilPlanner
from NetworkIO import TextureNetworkIO, DataSizeWithinStdOfMeanForProtoNetworkIO, \
    NoMoreThanXPercentDeviationPerProtoNetworkIO, FullConsensusEnsembleNetworkIO, VotingEnsembleNetworkIO

from typing import List, Tuple, Optional


def run_row_experiment(baseline_data: pd.DataFrame, data_size: int, data_texture: DataTextureEnum,
                       exfil_planner_factory: BaseExfilPlannerFactory,
                       network_io_factory: BaseNetworkIOFactory) -> pd.DataFrame:
    exfil_planner: BaseExfilPlanner = exfil_planner_factory(baseline_data)
    network_io: BaseNetworkIO = network_io_factory(baseline_data)

    exfil_data = ExfilData(urandom(data_size), data_texture)

    exfil_planner.set_network_io(network_io)
    exfil_planner.set_exfil_data(exfil_data)
    exfil_planner.reset()
    network_io.reset()

    actions_results_list: List[Tuple[Optional[Layer4Protocol], bool]] = exfil_planner.execute(True)
    actions, results = zip(*actions_results_list)

    total_result: bool = all(results)

    result_dict = dict(
        data_size=[data_size],
        data_texture=[data_texture.name],
        exfil_planner=[str(exfil_planner)],
        network_io=[str(network_io)],
        result=[total_result]
    )

    return pd.DataFrame(result_dict, index=['data_size', 'data_texture', 'exfil_planner', 'network_io', 'result'])


def wrap_row_experiment(baseline_data: pd.DataFrame, data_size: int, data_texture: DataTextureEnum,
                        exfil_planner_factory: BaseExfilPlannerFactory, network_io_factory: BaseNetworkIOFactory,
                        idx: int) -> Tuple[int, pd.DataFrame]:
    return idx, run_row_experiment(baseline_data, data_size, data_texture, exfil_planner_factory, network_io_factory)


def generate_grid_data(baseline_data: pd.DataFrame, data_sizes: List[int], data_textures: List[DataTextureEnum],
                       exfil_planners_factories: List[BaseExfilPlannerFactory],
                       network_ios_factories: List[BaseNetworkIOFactory], num_procs) -> pd.DataFrame:
    grid_permutations = itertools.product(data_sizes, data_textures, exfil_planners_factories, network_ios_factories)
    num_jobs = len(data_sizes) * len(data_textures) * len(exfil_planners_factories) * len(network_ios_factories)

    pbar = tqdm.tqdm(total=num_jobs, desc='Experiments')
    experiments_results: List[Optional[pd.DataFrame]] = [None] * num_jobs

    def update(index_result_tuple: Tuple[int, pd.DataFrame]):
        idx, result = index_result_tuple
        experiments_results[idx] = result  # put answer into correct index of result list
        pbar.update()

    with Pool(num_procs) as pool:
        results = [pool.apply_async(wrap_row_experiment, args=(baseline_data, *params, i), callback=update) for
                   i, (params) in enumerate(grid_permutations)]

        for res in results:
            res.wait()

    # experiments_results = [run_row_experiment(baseline_data, data_size, data_texture, exfil_planner, network_io) for
    #                        data_size, data_texture, exfil_planner, network_io in
    #                        tqdm.tqdm(grid_permutations, total=num_jobs, desc='Experiments')]

    return pd.concat(experiments_results, ignore_index=True)


def main():
    results_path = r"G:\ItzikProject\tmp_results"
    # sniff_results_path = path.join(results_path, 'sniff_results')
    sniff_results_path = results_path
    csv_name = "SkypeIRC_results.csv"

    num_procs: int = 8

    baseline_data: pd.DataFrame = pd.read_csv(path.join(sniff_results_path, csv_name), index_col=0)

    magnitudes: List[int] = [10 ** i for i in range(6, 12)]
    # mults: List[int] = list(range(1, 10))
    mults: List[int] = [1, 5]
    data_sizes: List[int] = sorted([mult * magnitude for mult, magnitude in itertools.product(magnitudes, mults)])

    data_textures: List[DataTextureEnum] = [val for val in DataTextureEnum]

    exfil_planners_factories: List[BaseExfilPlannerFactory] = [ExfilPlannerFactory(NaiveMaxDataProtocolExfilPlanner),
                                                               ExfilPlannerFactory(NaiveXPercentExfilPlanner),
                                                               ExfilPlannerFactory(
                                                                   NaiveProportionalWeightsRandomExfilPlanner,
                                                                   num_packets_for_split=1000)]

    network_ios_factories: List[BaseNetworkIOFactory] = [NetworkIOFactory(TextureNetworkIO),
                                                         NetworkIOFactory(DataSizeWithinStdOfMeanForProtoNetworkIO),
                                                         NetworkIOFactory(NoMoreThanXPercentDeviationPerProtoNetworkIO)]
    network_ios_factories_copy = network_ios_factories.copy()
    network_ios_factories.append(EnsembleNetworkIOFactory(VotingEnsembleNetworkIO, network_ios_factories_copy))
    network_ios_factories.append(EnsembleNetworkIOFactory(FullConsensusEnsembleNetworkIO, network_ios_factories_copy))

    grid_results: pd.DataFrame = generate_grid_data(baseline_data, data_sizes, data_textures, exfil_planners_factories,
                                                    network_ios_factories, num_procs)
    grid_results.to_csv(path.join(results_path, "planners_results.csv"), index=False)


if __name__ == "__main__":
    main()
