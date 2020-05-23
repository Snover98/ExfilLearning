import pandas as pd
from os import urandom
import itertools
from multiprocessing import Pool
import tqdm

from Protocols import Layer4Protocol
from ExfilData import ExfilData, DataTextureEnum
from ExfilPlanner import BaseExfilPlanner
from NetworkIO import BaseNetworkIO

from ExfilPlanner import NaiveMaxDataProtocolExfilPlanner, NaiveXPercentExfilPlanner, \
    NaiveProportionalWeightsRandomExfilPlanner
from NetworkIO import TextureNetworkIO, DataSizeWithinStdOfMeanForProtoNetworkIO, \
    NoMoreThanXPercentDeviationPerProtoNetworkIO

from typing import List, Tuple, Optional


def run_row_experiment(data_size: int, data_texture: DataTextureEnum, exfil_planner: BaseExfilPlanner,
                       network_io: BaseNetworkIO) -> pd.DataFrame:
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


def generate_grid_data(baseline_data: pd.DataFrame, data_sizes: List[int], data_textures: List[DataTextureEnum],
                       exfil_planners: List[BaseExfilPlanner], network_ios: List[BaseNetworkIO]) -> pd.DataFrame:
    for planner in exfil_planners:
        planner.set_baseline_data(baseline_data)

    for network_io in network_ios:
        network_io.set_baseline_data(baseline_data)

    grid_permutations = itertools.product(data_sizes, data_textures, exfil_planners, network_ios)
    num_jobs = len(data_sizes) * len(data_textures) * len(exfil_planners) * len(network_ios)

    experiments_results = [run_row_experiment(data_size, data_texture, exfil_planner, network_io) for
                           data_size, data_texture, exfil_planner, network_io in
                           tqdm.tqdm(grid_permutations, total=num_jobs, desc='Experiments')]

    return pd.concat(experiments_results, ignore_index=True)


def main():
    results_path = r"G:\ItzikProject\tmp_results"
    sniff_results_path = fr"{results_path}\sniff_results"

    baseline_data: pd.DataFrame = pd.read_csv(f"{sniff_results_path}\\bucket_16_results.csv", index_col=0)
    data_sizes: List[int] = [10 ** i for i in range(2, 7)]
    data_textures: List[DataTextureEnum] = [val for val in DataTextureEnum]
    exfil_planners: List[BaseExfilPlanner] = [NaiveMaxDataProtocolExfilPlanner(), NaiveXPercentExfilPlanner(),
                                              NaiveProportionalWeightsRandomExfilPlanner(num_packets_for_split=100)]
    network_ios: List[BaseNetworkIO] = [TextureNetworkIO(), DataSizeWithinStdOfMeanForProtoNetworkIO(),
                                        NoMoreThanXPercentDeviationPerProtoNetworkIO()]

    grid_results: pd.DataFrame = generate_grid_data(baseline_data, data_sizes, data_textures, exfil_planners,
                                                    network_ios)
    grid_results.to_csv(fr"{results_path}\planners_results.csv", index=False)


if __name__ == "__main__":
    main()
