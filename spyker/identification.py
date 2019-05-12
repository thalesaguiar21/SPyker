import sklearn.neural_network as skneural
import sklearn.model_selection as skselection


class _Identifier:

    def __init__(self, detector):
        self.detector = detector

    def identify(self, Xi):
        return self.detector.predict(Xi)


def create_identifier():
    pass


def train(X, y):
    loo = skselection.LeaveOneOut()
    best_acc, best_mlp = 0.0, None
    for mlp in _mlp_generator():
        hits = 0
        for train_index, test_index in loo.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            mlp.fit(X_train, y_train)
            if mlp.predict(X_test) == y_test:
                hits += 1
        accuracy = hits/X.shape[0] * 100
        if best_acc <= accuracy:
            best_mlp = mlp
    return _Identifier(best_mlp)


def _mlp_generator():
    """ Do a grid search over MLPClassifier """
    pass
