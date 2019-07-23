import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class BaseSelector:

    def __init__(self):
        self.estimator = None

    def fit(self, X, y):
        raise NotImplementedError()

    def transform(self, X, k):
        raise NotImplementedError()


class FtestSelector(BaseSelector):

    def __init__(self):
        super(FtestSelector, self).__init__()
        self.estimator = SelectKBest(score_func=f_classif)

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def transform(self, X, k):
        self.estimator.k = k
        return self.estimator.transform(X)


class MutualInformationSelector(BaseSelector):

    def __init__(self):
        super(MutualInformationSelector, self).__init__()
        self.estimator = SelectKBest(score_func=mutual_info_classif)

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def transform(self, X, k):
        self.estimator.k = k
        return self.estimator.transform(X)


class ChiSqSelector(BaseSelector):

    def __init__(self):
        super(ChiSqSelector, self).__init__()
        self.estimator = SelectKBest(score_func=chi2)

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def transform(self, X, k):
        self.estimator.k = k
        return self.estimator.transform(X)


class RandomForestSelector(BaseSelector):

    def __init__(self):
        super(RandomForestSelector, self).__init__()
        self.estimator = RandomForestClassifier()
        self.sorted_features = None

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.sorted_features = np.argsort(self.estimator.feature_importances_)[::-1]

    def transform(self, X, k):
        return X[:, self.sorted_features[:k]]


class LassoSelector(BaseSelector):

    def __init__(self):
        super(LassoSelector, self).__init__()
        self.estimator = LogisticRegression(penalty="l1")
        self.sorted_features = None

    def fit(self, X, y):
        self.estimator.fit(X, y)
        if self.estimator.coef_.ndim == 1:
            self.sorted_features = np.argsort(self.estimator.coef_)[::-1]
        else:
            importances = np.linalg.norm(self.estimator.coef_, axis=0, ord=1)
            self.sorted_features = np.argsort(importances)[::-1]

    def transform(self, X, k):
        return X[:, self.sorted_features[:k]]
