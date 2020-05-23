import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from SupervisedLearning.model_selection import *
from SupervisedLearning.standartisation import DFScaler
from SupervisedLearning.loading import *


class LogUniform:
    def __init__(self, low=0.0, high=1.0, size=None, base=10):
        self.low = low
        self.high = high
        self.size = size
        self.base = base

    def rvs(self, random_state):
        if random_state is None:
            state = np.random
        elif isinstance(random_state, np.random.RandomState):
            state = random_state
        else:
            state = np.random.RandomState(random_state)

        return np.power(self.base, state.uniform(self.low, self.high, self.size))


class RandIntMult:
    def __init__(self, low=0.0, high=1.0, mult=1, size=None):
        self.low = low
        self.high = high
        self.size = size
        self.mult = mult

    def rvs(self, random_state):
        if random_state is None:
            state = np.random
        elif isinstance(random_state, np.random.RandomState):
            state = random_state
        else:
            state = np.random.RandomState(random_state)

        return np.around(state.uniform(low=self.low, high=self.high, size=self.size) * self.mult).astype(int)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def find_best_models(train, valid, target: str, search_hyper_params=True, verbose=False):
    seed = np.random.randint(2 ** 31)
    print(f'seed is {seed}')
    print('')

    n_iter = 50

    random_forest_params = dict(
        n_estimators=RandIntMult(low=0.5, high=20.0, mult=100),
        criterion=['gini', 'entropy'],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=['auto', 'sqrt', 'log2', None],
        bootstrap=[True, False],
        max_depth=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    )

    svc_params = dict(
        C=LogUniform(low=-5.0, high=4.0),
        kernel=['linear', 'poly', 'rbf', 'sigmoid'],
        degree=[3, 4, 5],
        gamma=['scale', 'auto'],
        tol=LogUniform(low=-10, high=0),
        coef0=[0.0, 1.0]
    )

    ada_boost_params = dict(
        n_estimators=RandIntMult(low=0.5, high=20.0, mult=100),
        algorithm=['SAMME', 'SAMME.R'],
        learning_rate=LogUniform(low=-10, high=2)
    )

    knn_params = dict(
        n_neighbors=RandIntMult(low=0.5, high=20.0, mult=10),
        weights=['uniform', 'distance'],
        algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
        p=[1, 2]
    )

    estimators = [
        RandomForestClassifier(n_jobs=1),
        AdaBoostClassifier(),
        SVC(probability=True, max_iter=60000),
        KNeighborsClassifier()
    ]
    params = [random_forest_params,
              ada_boost_params,
              svc_params,
              knn_params]

    eval_func_names = ['Precision', 'Recall', 'Balanced Accuracy', 'F1']
    eval_funcs = [evaluate_precision_score, evaluate_recall_score, evaluate_balanced_accuracy_score, evaluate_f1_score]

    eval_func_names = ['Balanced Accuracy']
    eval_funcs = [evaluate_balanced_accuracy_score]

    # eval_func = evaluate_precision_score
    #
    # best_estimator = find_problem_best_model(train, valid, estimators, params, eval_func, n_iter, seed,
    #                                          search_hyper_params, verbose)

    best_estimators = [
        find_problem_best_model(train, valid, estimators, params, eval_func, eval_func_name, target, n_iter, seed,
                                search_hyper_params, verbose, add_voting_clf=True)
        for eval_func, eval_func_name in zip(eval_funcs, eval_func_names)
    ]

    return best_estimators


def use_estimators(best_estimators, train, valid, test, target: str):
    eval_funcs = [precision_score, recall_score, balanced_accuracy_score, f1_score]
    eval_func_names = ['Precision', 'Recall', 'Balanced Accuracy', 'F1']
    # best_precision, best_recall, best_balanced_accuracy, best_f1 = best_estimators

    eval_funcs = [balanced_accuracy_score]
    eval_func_names = ['Balanced Accuracy']

    features = list(set(train.columns.to_numpy().tolist()).difference({target}))

    non_test_data = pd.concat((train, valid))

    for estimator, eval_func, eval_func_name in zip(best_estimators, eval_funcs, eval_func_names):
        # data for the final classifier
        non_test_data_for_model = model_train_set(estimator, non_test_data, target)
        print('')
        print('============================================')
        estimator.fit(*target_features_split(non_test_data_for_model, target))
        test_pred = pd.Series(estimator.predict(test[features]), index=test.index)
        test_true = test[target]
        print(f"The best {eval_func_name} model has a score of: {eval_func(test_true, test_pred)}")
        plot_confusion_matrix(test_true, test_pred, [False, True], title=f'Best {eval_func_name} Confusion Matrix')
        eval_func_name_file_format: str = eval_func_name.lower().replace(' ', '_')
        print('')
        test_pred.to_csv(f'test_{eval_func_name_file_format}_predictions.csv', index=False)


def main():
    results_path: str = r"G:\ItzikProject\tmp_results"
    target = 'result'

    df = load_csv(fr"{results_path}\planners_results.csv")
    df = change_categoricals(df, target)
    # change 'data_size' to the log scale
    df['data_size_log'] = np.log(df['data_size'])
    df = df.drop(columns=['data_size'])

    train, valid, test = split_data(df, target)

    scaler = DFScaler(train, ['data_size_log'])

    train = scaler.scale(train)
    valid = scaler.scale(valid)
    test = scaler.scale(test)

    verbose = True
    search_hyper_params = True

    best_models = find_best_models(train, valid, target, verbose=verbose, search_hyper_params=search_hyper_params)
    use_estimators(best_models, train, valid, test, target)


if __name__ == '__main__':
    main()
