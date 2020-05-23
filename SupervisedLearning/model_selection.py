import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.utils import resample
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score
from numpy.linalg import norm
from .hyper_params_funcs import *


def evaluate_balanced_accuracy_score(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)
    return balanced_accuracy_score(y_true, y_pred)


def evaluate_recall_score(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)

    return recall_score(y_true, y_pred)


def evaluate_precision_score(estimator, X, y_true) -> float:
    y_pred = estimator.predict(X)

    return precision_score(y_true, y_pred)


def evaluate_f1_score(estimator, X, y_true):
    """
    evaluate the likely voters problem on a specific party
    :param indices_pred:
    :param y_true:
    :return:
    """
    y_pred = estimator.predict(X)
    return f1_score(y_true, y_pred, )


def upsample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    balances a dataframe according to a target label
    :param df: the dataframe
    :param target: the target label
    :return: the dataframe, now balanced according to the target values
    """
    targets = df[target]
    classes = targets.unique()

    # number of appearances per class
    num_appearances = {target_class: targets[targets == target_class].size for target_class in classes}
    # basically argmax
    most_common = max(num_appearances.keys(), key=(lambda key: num_appearances[key]))

    # upsampled sub-dataframes for the values that aren't the most common
    minority_upsampled = [resample(df[targets == target_class], replace=True, n_samples=num_appearances[most_common])
                          for target_class in classes if target_class != most_common]

    # balanced dataframe
    df_upsampled = pd.concat([df[targets == most_common]] + minority_upsampled)

    return df_upsampled


def target_features_split(df: pd.DataFrame, target: str):
    """
    splits the dataframe to features and target label (X and y)
    :param df: the dataframe
    :param target: the target label
    :return: 2 dataframes, the first with only the features, the second with only the labels
    """
    features = list(set(df.columns.to_numpy().tolist()).difference({target}))
    return df[features], df[target]


def cross_valid(model, df: pd.DataFrame, num_folds: int, eval_func, target):
    """
    peforms k-fold cross validation on the dataframe df according to the evaluation function
    :param model: the model we are checking
    :param df: the dataframe we want to cross validate
    :param num_folds: the number of folds
    :param eval_func: the evaluation function
    :param target: the target label
    :return: the average score across the folds
    """
    kf = StratifiedKFold(n_splits=num_folds)
    score = 0

    df_features, df_targets = target_features_split(df, target)

    for train_indices, test_indices in kf.split(df_features, df_targets):
        train_targets, train_features = df_targets[train_indices], df_features.iloc[train_indices]
        test_targets, test_features = df_targets[test_indices], df_features.iloc[test_indices]

        model.fit(train_features, train_targets)
        score += eval_func(model, test_features, test_targets)

    return score / num_folds


def choose_best_model(models, valid: pd.DataFrame, eval_func, eval_func_name: str, target: str, verbose: bool = False):
    """
    return the model that performs the best on the validation set according to the evaluation function
    :param models: list of models to check, already fitterd on the training set
    :param valid: the validation set
    :param eval_func: the evaluation function for the problem
    :param eval_func_name: the name of the evaluation function for the problem
    :param target: the target label
    :param verbose: verbose flag
    :return: the best performing estimator
    """
    best_score = -np.inf
    best_model = None

    # train_features, train_targets = target_features_split(train, target)
    valid_features, valid_targets = target_features_split(valid, target)

    for model in models:
        # model.fit(train_features, train_targets)
        score = eval_func(model, valid_features, valid_targets)

        if verbose:
            print(f'Model {get_model_name(model)} has a {eval_func_name} score of {score}')

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


def wrapper_params(params: dict):
    """
    converts the inputted params dictionary into ones that fit with the wrappers
    :param params: parameters dictionary (each key is a hyper-paramter)
    :return: converted params to fit the wrappers
    """
    return {'model__' + key: value for key, value in params.items()}


def is_model_balanced(model) -> bool:
    """
    :param model: the model that checked
    :return: True if the model balances the class weight of the labels, False otherwise
    """
    params = get_normal_params(model.get_params())
    return 'class_weight' in params.keys() and params['class_weight'] == 'balanced'


def model_train_set(model, train, target):
    """
    function that returns a balanced training set if the model does not balance on it's own
    :param model: the model
    :param train: the actual training set
    :param target: the target label
    :return: if the model already balances on it's own - train, otherwise a balanced version of train
    """
    return train if is_model_balanced(model) else upsample(train, target)


def choose_hyper_params(models, params_ranges, eval_func, train, target, num_folds=5,
                        random_state=None, n_iter=10, verbose=False):
    """
    given a list of models and an evaluation function, find the best hyper parameters for each model to get the best score
    :param models: list of models
    :param params_ranges: parameter distributions for each estimator
    :param eval_func: evaluation function for the problem
    :param train: training set
    :param target: the target label
    :param num_folds: the number of folds used in cross-validation
    :param random_state: random state (int or np.RandomState) to use for the randomized search
    :param n_iter: number of iterations for the randomized search on each estimator
    :param verbose: verbose flag
    :return: the estimators with tuned hyper parameters fitted on the training set
    """

    best_models = []
    for model, params in zip(models, params_ranges):
        if verbose:
            print(f'Tuning model #{len(best_models) + 1}: {get_model_name(model)}')

        grid = RandomizedSearchCV(model, params, scoring=eval_func, cv=num_folds, random_state=random_state,
                                  n_iter=n_iter, n_jobs=-1)

        grid.fit(*target_features_split(model_train_set(model, train, target), target))
        best_models.append(grid.best_estimator_)

    return best_models


def find_problem_best_model(train, valid, estimators, params, eval_func, eval_func_name: str, target: str, n_iter=10,
                            seed=None, search_hyper_params=True, verbose=False, add_voting_clf=False):
    """
    :param train: train set
    :param valid: validiation set
    :param estimators: estimators that we should
    :param params: parameter distributions for each estimator
    :param eval_func: evaluation function for said problem
    :param eval_func_name: the name of the evaluation function
    :param target: the target label
    :param n_iter: number of random guesses per estimator
    :param seed: random seed for random search
    :param search_hyper_params: boolean flag for whether we want to do a random search or load from file
    :param verbose: flag for verbose prints
    :param add_voting_clf: flag for adding a voting classifier of the best classifiers
    :return: the best performing estimator for the problem
    """
    print('============================================')
    print(f'started evaluation')
    if search_hyper_params:
        best_estimators = choose_hyper_params(estimators, params, eval_func, train, target, random_state=seed,
                                              n_iter=n_iter, verbose=verbose)
        save_problem_hyper_params(best_estimators, eval_func_name)
    else:
        best_estimators = load_problem_hyper_params(estimators, eval_func_name, verbose=verbose)
        for estimator in best_estimators:
            estimator_train = model_train_set(estimator, train, target)
            estimator.fit(*target_features_split(estimator_train, target))

    if add_voting_clf:
        estimators_with_names = [(get_model_name(estimator), estimator) for estimator in best_estimators]
        voting_estimator = VotingClassifier(estimators_with_names, voting='soft')
        estimator_train = model_train_set(voting_estimator, train, target)
        voting_estimator.fit(*target_features_split(estimator_train, target))
        best_estimators.append(voting_estimator)

    print_best_hyper_params(best_estimators, eval_func_name)
    best_estimator = choose_best_model(best_estimators, valid, eval_func, eval_func_name, target, verbose=verbose)
    print_best_model(best_estimator, eval_func_name)

    return best_estimator
