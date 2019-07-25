import abc
import typing
from pandas import DataFrame
from alphaml.engine.components.data_manager import DataManager


"""
Pipeline Framework:
  imputer_operator[1] ==> encoder and label encoder[1] ==> normalizer[0]
  ==> feature_generation_operator[0] ==> feature_selection_operator[0]
"""


class DP_Pipeline(object):
    def __init__(self, pipeline_config):
        """Reconstruct the data preprocessing pipeline according to the config."""
        self.pipeline_operators = list()

    def execute(self, input: DataFrame) -> DataManager:
        input_data = [input]
        for operator in self.pipeline_operators:
            # Prepare the input the operator.
            input_data = operator.operate(input_data)
        return input_data
