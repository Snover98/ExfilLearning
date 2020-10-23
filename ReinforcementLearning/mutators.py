import collections

import numpy as np
import pandas as pd

from typing import List, Callable, Union, Sequence

Mutator = Callable[[pd.DataFrame], pd.DataFrame]


def mutate_baselines(baseline_datas: List[pd.DataFrame],
                     baselines_mutators: List[Mutator],
                     inplace_mutations: Union[bool, Sequence[bool]] = True) -> List[pd.DataFrame]:
    """
    Mutates the inputted baselines using the mutators.
    A mutation can be either inplace or create a new mutated version of each baselines in addition to the original
    This is done using the sequence inplace_mutations - for each mutator if the boolean in the same position within
    the sequence is True the mutation will be inplace, and if False the mutation will create a copy.
    If inplace_mutations is a single boolean, it will be treated as if all mutators have that value

    :param baseline_datas: the baselines to mutate
    :param baselines_mutators: the mutators to use - each will be used an all baselines
    :param inplace_mutations: marks which mutators will be inplace and which will create copies
    :return: a list of all mutated baselines (can contain both inplace and new mutations)
    """
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
    """
    multiplies the total amount of bytes and packets by a constant integer
    :param mult: the multitude by which the sizes will be multiplied
    :param baseline_data: the baseline to be mutated
    :return: a copy of the data with the sizes multiplied by the constant
    """
    mutated_baseline = baseline_data.copy()
    mutated_baseline[['total_bytes', 'num_packets']] *= mult
    return mutated_baseline


def sizes_mult_mutator(mult: int) -> Mutator:
    """
    creates a mutation method of `mult_baseline_sizes_mutation` with the inputted value
    :param mult: the constant the `mult_baseline_sizes_mutation` will use
    :return: the mutation method
    """
    return lambda baseline_data: mult_baseline_sizes_mutation(mult, baseline_data)


def shuffle_protocols_mutation(baseline_data: pd.DataFrame) -> pd.DataFrame:
    """
    shuffle the indices of a baseline randomly
    :param baseline_data: the baseline to mutate
    :return: a copy of the baseline with the indices (protocols) shuffled
    """
    shuffled_indices: list = baseline_data.index.to_numpy().tolist()
    np.random.shuffle(shuffled_indices)

    return pd.DataFrame(baseline_data.to_numpy(), columns=baseline_data.columns, index=shuffled_indices)


def switch_2_protocols_mutation(baseline_data: pd.DataFrame) -> pd.DataFrame:
    """
    switches the indices of 2 randomly chosen rows in the baseline
    :param baseline_data: the baseline to mutate
    :return: a copy of the baseline with 2 rows' indices being switched
    """
    indices: list = baseline_data.index.to_numpy().tolist()

    idx_to_switch = np.random.choice(np.arange(len(indices)), size=2, replace=False)
    indices[idx_to_switch[0]], indices[idx_to_switch[1]] = indices[idx_to_switch[1]], indices[idx_to_switch[0]]

    return pd.DataFrame(baseline_data.to_numpy(), columns=baseline_data.columns, index=indices)


def change_values_by_x_percent_mutation(baseline_data: pd.DataFrame, mutation_percentage: float,
                                        features_to_change: List[str] = None,
                                        protocols_to_change: List[str] = None) -> pd.DataFrame:
    """
    randomly changes the features of a baseline randomly by at most mutation_percentage,
    only in the protocols and features specified.
    Each value will that will be changed will be multiplied by a random number in the range (1-X, 1+X),
    When X is the mutation percentage
    :param baseline_data: the baseline to mutate
    :param mutation_percentage: the max percentage by which the mutated values can change from the original
    :param features_to_change: the features to be changed randomly. by default all features are changed
    :param protocols_to_change: the protocols to be changed randomly. by default all protocols are changed
    :return: the baseline randomly changed
    """
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
    """
    Returns a `change_values_by_x_percent_mutation` with the inputted percentage, features and protocols
    :param mutation_percentage: the max percentage by which the mutated values can change from the original
    :param features_to_change: the features to be changed randomly. by default all features are changed
    :param protocols_to_change: the protocols to be changed randomly. by default all protocols are changed
    :return: the mutation method
    """
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
