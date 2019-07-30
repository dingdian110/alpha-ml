import typing
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.base_operator import Operator, FEATURE_SELECTION


class NaiveSelectorOperator(Operator):
    def __init__(self, params=[50, 0], metric=None):
        '''
        :param params: A list. The first element stands for k in 'k-best', the second stands for metric function
                        0 for chi2, 1 for f_classif, 2 for mutual_info_classif, 3 for f_regression, 4 for mutual_info_regression
                        0,1,2 are for classification and 0,3,4 are for regression
                        Warnings: function chi2 can be used only if the features are non-negative
        :param metric: Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
        '''
        super().__init__(FEATURE_SELECTION, 'fs_kbestselector', params)
        k_best, metric = params
        if metric == 0:
            self.metric = chi2
        elif metric == 1:
            self.metric = f_classif
        elif metric == 2:
            self.metric = mutual_info_classif
        elif metric == 3:
            self.metric = f_regression
        elif metric == 4:
            self.metric = mutual_info_regression
        elif not callable(metric):
            raise ValueError("Expected callable metric function for K-best Selector operator!")
        else:
            self.metric = metric

        self.k_best = k_best
        self.selector = SelectKBest(self.metric, k_best)

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a NaiveSelectorOperator is a list of DataManager
        self.check_phase(phase)

        x = None
        y = None
        if phase == 'train':
            for dm in dm_list:
                if x is None:
                    x = dm.train_X
                    y = dm.train_y
                else:
                    x = np.hstack((x, dm.train_X))
            x = self.selector.fit_transform(x, y)
            dm = DataManager(x, y, spilt=False)
        else:
            for dm in dm_list:
                if x is None:
                    x = dm.test_X
                else:
                    x = np.hstack((x, dm.test_X))
            x = self.selector.transform(x)
            dm = DataManager()
            dm.test_X = x
        return dm


class MLSelectorOperator(Operator):
    RANDOM_FOREST = 0
    LASSO_REGRESSION = 1

    CLASSIFICATION = 0
    REGRESSION = 1

    def __init__(self, params=[50, 0, 0]):
        '''
        :param params:A list. The first element stands for k in 'k-best'
                        The second element stands for task type (0 for classification and 1 for regression)
                        The three element stands for machine learning models:
                        0 for RandomForest, 1 for LassoRegression
        '''
        super().__init__(FEATURE_SELECTION, 'fs,mlselector', params)
        self.kbest, self.task, self.model = params
        if self.model == self.RANDOM_FOREST:
            if self.task == self.CLASSIFICATION:
                from sklearn.ensemble import RandomForestClassifier
                self.selector = RandomForestClassifier()
            elif self.task == self.REGRESSION:
                from sklearn.ensemble import RandomForestRegressor
                self.selector = RandomForestRegressor()
        elif self.model == self.LASSO_REGRESSION:
            if self.task == self.CLASSIFICATION:
                from sklearn.linear_model import LogisticRegression
                self.selector = LogisticRegression(penalty='l1')
            elif self.task == self.REGRESSION:
                from sklearn.linear_model import Lasso
                self.selector = Lasso()
        self.sorted_features = None

    def operate(self, dm_list: typing.List, phase='train'):
        '''
        :return: self.result_dm is a new Datamanager with data splited for training and validation
        '''
        x = None
        y = None
        if phase == 'train':
            for dm in dm_list:
                if x is None:
                    x = dm.train_X
                    y = dm.train_y
                else:
                    x = np.hstack((x, dm.train_X))
            self.selector.fit(x, y)
        else:
            for dm in dm_list:
                if x is None:
                    x = dm.test_X
                else:
                    x = np.hstack((x, dm.test_X))

        if self.model == self.RANDOM_FOREST:
            self.sorted_features = np.argsort(self.selector.feature_importances_)[::-1]
        elif self.model == self.LASSO_REGRESSION:
            if self.selector.coef_.ndim == 1:
                self.sorted_features = np.argsort(self.selector.coef_)[::-1]
            else:
                importances = np.linalg.norm(self.selector.coef_, axis=0, ord=1)
                self.sorted_features = np.argsort(importances)[::-1]
        x = x[:, self.sorted_features[:self.kbest]]
        dm = DataManager()
        if phase == 'train':
            dm.train_X = x
            dm.train_y = y
        else:
            dm.test_X = x
        return dm
