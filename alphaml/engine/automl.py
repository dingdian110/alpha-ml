import numpy as np


class AutoML(object):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, task: int):
        return self

    def predict(self, X):
        return None

    def score(self, X, y):
        return None


class AutoMLClassifier(AutoML):
    pass
