import typing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.base_operator import Operator, FEATURE_GENERATION
from alphaml.engine.components.feature_engineering.auto_cross import AutoCross


class PolynomialFeaturesOperator(Operator):
    def __init__(self, params=2):
        '''
        :param params: Stand for degrees. Default 2
        '''
        super().__init__(FEATURE_GENERATION, 'fg_polynomial', params)
        self.polynomialfeatures = PolynomialFeatures(degree=params, interaction_only=True)

    def operate(self, dm_list: typing.List, phase='train') -> DataManager:
        # The input of a PolynomialFeatureOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numericial_index = [i for i in range(len(feature_types))
                            if feature_types[i] == "Float" or feature_types[i] == "Discrete"]
        init_length = len(numericial_index) + 1
        if phase == 'train':
            x = dm.train_X
            newfeatures = self.polynomialfeatures.fit_transform(x[:, numericial_index])
            dm.train_X = newfeatures[:, init_length:]
        else:
            x = dm.test_X
            newfeatures = self.polynomialfeatures.transform(x[:, numericial_index])
            dm.test_X = newfeatures[:, init_length:]
        return dm


class AutoCrossOperator(Operator):
    def __init__(self, stratify, metric=None, params=50):
        super().__init__(FEATURE_GENERATION, 'fg_autocross', params)
        self.metric = metric
        self.autocross = AutoCross(max_iter=params, stratify=stratify, metrics=metric)
        assert isinstance(stratify, bool)
        self.stratify = stratify

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a AutoCrossOperator is a DataManager
        assert len(dm_list) == 1
        dm = dm_list[0]
        assert isinstance(dm, DataManager)
        self.check_phase(phase)

        feature_types = dm.feature_types
        onehot_index = [i for i in range(len(feature_types))
                        if feature_types[i] == "One-Hot"]
        numerical_index = [i for i in range(len(feature_types))
                           if feature_types[i] == 'Discrete' or feature_types[i] == 'Float']

        if phase == 'train':
            from sklearn.model_selection import train_test_split
            if self.stratify:
                train_x, val_x, train_y, val_y = train_test_split(dm.train_X, dm.train_y, test_size=0.2,
                                                                  stratify=dm.train_y)
            else:
                train_x, val_x, train_y, val_y = train_test_split(dm.train_X, dm.train_y, test_size=0.2)
            x = dm.train_X
            self.autocross.fit(train_x, val_x, train_y, val_y, onehot_index, numerical_index)
            dm.train_X = self.autocross.transform(x)

        else:
            x = dm.test_X
            dm.test_X = self.autocross.transform(x)
        return dm


class PCAOperator(Operator):
    def __init__(self, params=10):
        '''
        :param params: Stand for n_components. Default 10
        '''
        super().__init__(FEATURE_GENERATION, 'fg_pca', params)
        self.pca = PCA(whiten=True, n_components=params)

    def operate(self, dm_list: typing.List, phase='train'):
        # The input of a PCAOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        self.check_phase(phase)

        dm = dm_list[0]
        feature_types = dm.feature_types
        numerical_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]
        if phase == 'train':
            x = dm.train_X
            dm.train_X = self.pca.fit_transform(x[:, numerical_index])
        else:
            x = dm.test_X
            dm.test_X = self.pca.fit_transform(x[:, numerical_index])
        return dm
