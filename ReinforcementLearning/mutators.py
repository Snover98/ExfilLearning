import collections

import numpy as np
import pandas as pd

from typing import List, Callable, Union, Sequence

Mutator = Callable[[pd.DataFrame], pd.DataFrame]


def mutate_baselines(baseline_datas: List[pd.DataFrame],
                     baselines_mutators: List[Mutator],
                     inplace_mutations: Union[bool, Sequence[bool]] = True) -> List[pd.DataFrame]:
    if isinstance(inplace_mutations, collections.Sequence):
        assert len(inplace_mutations) == len(baselines_mutators), "inplace specifications must match number of mutators"
    else:
        inplace_mutations: List[bool] = [inplace_mutations] * len(baselines_mutators)

    mutated_baselines = [baseline_data.copy() for baseline_data in baseline_datas]

    for mutator, is_inplace_mutation in zip(baselines_mutators, inplace_mutations):
        if is_inplace_mutation:
            mutated_baselines = [mutator(baseline_data) for baseline_data in mutated_baselines]
        else:
            mutated_baselines += [mutator(baseline_data) for baseline_data in mutated_baselines]

    return mutated_baselines


def mult_baseline_sizes_mutation(mult: int, baseline_data: pd.DataFrame) -> pd.DataFrame:
    mutated_baseline = baseline_data.copy()
    mutated_baseline[['total_bytes', 'num_packets']] *= mult
    return mutated_baseline


def sizes_mult_mutator(mult: int) -> Mutator:
    return lambda baseline_data: mult_baseline_sizes_mutation(mult, baseline_data)


def shuffle_protocols_mutation(baseline_data: pd.DataFrame) -> pd.DataFrame:
    shuffled_indices = baseline_data.index.to_numpy().tolist()
    np.random.shuffle(shuffled_indices)

    return pd.DataFrame(baseline_data.to_numpy(), columns=baseline_data.columns, index=shuffled_indices)


def switch_2_protocols_mutation(baseline_data: pd.DataFrame) -> pd.DataFrame:
    indices = baseline_data.index.to_numpy().tolist()

    idx_to_switch = np.random.choice(np.arange(len(indices)), size=2, replace=False)
    indices[idx_to_switch[0]], indices[idx_to_switch[1]] = indices[idx_to_switch[1]], indices[idx_to_switch[0]]

    return pd.DataFrame(baseline_data.to_numpy(), columns=baseline_data.columns, index=indices)


def change_values_by_x_percent_mutation(baseline_data: pd.DataFrame, mutation_percentage: float,
                                        features_to_change: List[str] = None,
                                        protocols_to_change: List[str] = None) -> pd.DataFrame:
    if features_to_change is None:
        features_to_change = baseline_data.columns.to_numpy().tolist()

    if protocols_to_change is None:
        protocols_to_change = baseline_data.index.to_numpy().tolist()

    rand_vals = pd.DataFrame(
        np.random.uniform(1 - mutation_percentage, 1 + mutation_percentage, size=baseline_data.to_numpy().shape),
        columns=baseline_data.columns, index=baseline_data.index)

    # set features and protocols that should not be changed to be multiplied by 1
    rand_vals[list(set(rand_vals.columns).difference(features_to_change))] = 1.0
    rand_vals.loc[list(set(rand_vals.index).difference(protocols_to_change))] = 1.0

    mutated_data = baseline_data * rand_vals
    for col in mutated_data.columns:
        mutated_data[col] = mutated_data[col].astype(baseline_data[col].dtype)

    return mutated_data


def rand_by_x_percent_mutator(mutation_percentage: float, features_to_change: List[str] = None,
                              protocols_to_change: List[str] = None) -> Mutator:
    return lambda baseline_data: change_values_by_x_percent_mutation(baseline_data, mutation_percentage,
                                                                     features_to_change, protocols_to_change)


__all__ = [
    'Mutator',
    'mutate_baselines',
    'mult_baseline_sizes_mutation',
    'sizes_mult_mutator',
    'shuffle_protocols_mutation',
    'switch_2_protocols_mutation',
    'change_values_by_x_percent_mutation',
    'rand_by_x_percent_mutator',
]
