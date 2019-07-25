import abc
import typing
from pandas import DataFrame
from alphaml.engine.components.data_manager import DataManager


class Operator(object, metaclass=abc.ABCMeta):
    def __init__(self, type, operator_name, params=None):
        self.type = type   # type: "data_preprocessing", "feature_generation", "feature_selection".
        self.operator_name = operator_name  # id: dp_minmaxnormalizer, fg_polynomial_features, fs_lasso.
        self.params = params
        self.id = None
        self.origins = None
        self.result_dm = None

    @abc.abstractmethod
    def operate(self, dm_list: typing.List) -> DataManager:
        # After this operator, gc the result of operator.
        pass


class EmptyOperator(Operator):
    def __init__(self):
        super().__init__('Empty', 'empty_operatpr')

    def operate(self, dm_list: typing.List[DataManager]):
        self.result_dm = dm_list[-1]
