import typing

from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.base_operator import Operator, FEATURE_GENERATION


class PolynomialFeatureOperator(Operator):
    def __init__(self, params=None):
        super().__init__(FEATURE_GENERATION, 'fg_polynomial', params)

    def operate(self, dm_list: typing.List) -> DataManager:
        # The input of a PolynomialFeatureOperator is a DataManager
        assert len(dm_list) == 1 and isinstance(dm_list[0], DataManager)
        dm = dm_list[0]
