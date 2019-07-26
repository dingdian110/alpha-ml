import typing
from sklearn.preprocessing import PolynomialFeatures

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.base_operator import Operator, FEATURE_GENERATION


class PolynomialFeaturesOperator(Operator):
    def __init__(self, params=2):
        '''
        :param params: Stand for degree. Default 2
        '''
        super().__init__(FEATURE_GENERATION, 'fg_polynomial', params)
        self.polynomialfeatures = PolynomialFeatures(degree=params, interaction_only=True)

    def operate(self, dm_list: typing.List) -> DataManager:
        # The input of a PolynomialFeatureOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
        feature_types = dm.feature_types
        numercial_index = [i for i in range(len(feature_types))
                           if feature_types[i] == "Float" or feature_types[i] == "Discrete"]
        x = dm.train_X
        init_length = len(numercial_index) + 1
        polynomialfeatures = self.polynomialfeatures
        newfeatures = polynomialfeatures.fit_transform(x[:, numercial_index])
        dm.train_X = newfeatures[:, init_length:]
        self.result_dm = dm


class AutoCrossOperator(Operator):
    def __init__(self, params=50, metric=None):
        super().__init__(FEATURE_GENERATION, 'fg_autocross', params)
        if metric is None:
            raise ValueError("Expected metric function for auto-cross operator!")
        elif not callable(metric):
            raise ValueError("Expected callable metric function for auto-cross operator!")
        else:
            self.metric = metric

    def operate(self, dm_list: typing.List):
        pass
