import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# def find_worst_feature(data, labels, clf, selected):
#     best_perf = -1
#     worst_feature = None
#     for f in selected:
#         selected_tmp = list(selected)
#         selected_tmp.remove(f)
#         if len(selected_tmp) == 0:
#             perf = 0
#         else:
#             perf = _evaluate(data, labels, clf, "holdout", selected_tmp)
#         if perf > best_perf:
#             worst_feature = f
#             best_perf = perf
#
#     return worst_feature, best_perf
#
#
# def sffs(data, labels, d, clf, metrics):
#     # initialization
#     X = [[]] * (d + 2)
#     Jx = [0] * (d + 2)
#     Y = list(range(data.shape[1]))
#     k = 0
#     while k < d:
#         # find the most significant feature
#         best_perf = -1
#         best_feature = None
#         remain = [feat for feat in Y if feat not in X[k]]
#
#         for f in remain:
#             selected_tmp = list(X[k])
#             selected_tmp.append(f)
#             perf = _evaluate(data, labels, clf, "holdout", selected_tmp)
#             if perf > best_perf:
#                 best_feature = f
#                 best_perf = perf
#
#         X[k + 1] = list(X[k])
#         X[k + 1].append(best_feature)
#         Jx[k + 1] = best_perf
#         k = k + 1
#         # find the least significant feature
#         worst_feature, best_perf = find_worst_feature(data, labels, clf, X[k])
#         while best_perf > Jx[k - 1]:
#             X[k - 1] = list(X[k])
#             X[k - 1].remove(worst_feature)
#             Jx[k - 1] = best_perf
#
#             k = k - 1
#             worst_feature, best_perf = find_worst_feature(data, labels, clf, X[k])
#
#         # print("k:", k, "Jx:", Jx[k], "time:", time() - t0)
#
#     return X[d]
#
#
# def sfs(x, y, k, clf, metrics):
#     feature_num = x.shape[1]
#     selected = []
#     remain = list(range(feature_num))
#
#     while len(selected) < k:
#         best_perf = -1
#         best_feature = None
#         for f in remain:
#             selected_tmp = list(selected)
#             selected_tmp.append(f)
#             perf = _evaluate(x, y, clf, "holdout", selected_tmp)
#             if perf > best_perf:
#                 best_perf = perf
#                 best_feature = f
#         selected.append(best_feature)
#         remain.remove(best_feature)
#         # print("features =", len(selected), "perf =", best_perf, "time =", time() - t0)
#
#     return selected


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
