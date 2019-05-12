import sklearn.neural_network as skneural
import sklearn.model_selection as skselection
import numpy as np
import pdb
from itertools import product
from enum import Enum


class _Identifier:

    def __init__(self, detector):
        self.detector = detector

    def identify(self, Xi):
        return self.detector.predict(Xi)


def create_trained_identifier(X, y):
    best_accuracy = 0
    best_mlp = None
    curr_mlp = 0
    for mlp in _mlp_generator():
        curr_mlp += 1
        msg = 'Training and testing MLP for row {} from grid...'
        print(msg.format(curr_mlp), end='\r')
        if train_and_test(mlp, X, y) > best_accuracy:
            best_mlp = mlp
    print('Best MLP achieved', best_accuracy, 'accuracy!')
    return _Identifier(best_mlp)


def train_and_test(mlp, X, y):
    loo = skselection.LeaveOneOut()
    hits = 0
    for train_index, test_index in loo.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        mlp.fit(X_train, y_train)
        if mlp.predict(X_test) == y_test:
            hits += 1
    accuracy = hits/X.shape[0] * 100
    return accuracy


def _mlp_generator():
    """ Do a grid search over MLPClassifier """
    hidden_layer_step = [(_MIN_HL_SIZE + t,) for t in range(0, 201, 50)]
    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs', 'sgd', 'adam']
    learnrate = [0.0001, 0.001, 0.01]
    max_iters = [x for x in range(100, 501, 100)]
    tols = [0.0001, 0.001, 0.01]
    n_iter_no_changes = [x for x in range(5, 31, 5)]
    grid = product(hidden_layer_step, activations, solvers, learnrate,
                   max_iters, tols, n_iter_no_changes)
    for row in grid:
        mlp = skneural.MLPClassifier(
                hidden_layer_sizes=row[GRIDIndexes.HLSIZE],
                activation=row[GRIDIndexes.ACTIVATION],
                solver=row[GRIDIndexes.SOLVER],
                alpha=row[GRIDIndexes.ALPHA],
                max_iter=row[GRIDIndexes.MAXITER],
                tol=row[GRIDIndexes.TOL],
                n_iter_no_change=row[GRIDIndexes.NITER])
        yield mlp


_MIN_HL_SIZE = 100


class GRIDIndexes(Enum):
    HLSIZE= 0
    ACTIVATION = 1
    SOLVER = 2
    ALPHA = 3
    MAXITER = 4
    TOL = 5
    NITER = 6
