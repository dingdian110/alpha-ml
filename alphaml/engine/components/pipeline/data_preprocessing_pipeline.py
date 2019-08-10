import abc
import typing
from pandas import DataFrame
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.components.pipeline.data_preprocessing_operator import *
from alphaml.engine.components.pipeline.feature_generation_operator import *
from alphaml.engine.components.pipeline.feature_selection_operator import *

"""
Pipeline Framework:
  imputer_operator[1] ==> encoder and label encoder[1] ==> normalizer[0]
  ==> feature_generation_operator[0] ==> feature_selection_operator[0]
"""


class DP_Pipeline(object):
    def __init__(self, pipeline_config):
        """Reconstruct the data preprocessing pipeline according to the config."""
        # Create the default DP graph.
        # 1. Create the basic nodes.
        self.pipeline_operators = list()
        node1 = ImputerOperator()
        node2 = LabelEncoderOperator()
        node3 = FeatureEncoderOperator()
        # node4 = NormalizerOperator()
        # node5 = ScalerOperator()
        node6 = ConstantRemoverOperator()
        node7 = VarianceRemoverOperator()
        node8 = ZeroOperator()
        node9 = PCAOperator()
        node10 = IdenticalOperator()

        self.pipeline_operators.extend([node1, node2, node3, node6, node7, node8, node10])
        self.cached_dm = dict()

        # Assign the node id.
        last_dp_operator_id = 0
        fg_operator_list = []
        for node_id, node in enumerate(self.pipeline_operators):
            node.id = node_id
            if 'dp' in node.operator_name:
                last_dp_operator_id = node_id
            if 'fg' in node.operator_name:
                fg_operator_list.append(node_id)
            if node_id != 0:
                node.origins = [node_id - 1]
            if 'fg' in node.operator_name:
                node.origins = [last_dp_operator_id]
            if 'fs' in node.operator_name:
                node.origins = fg_operator_list + [last_dp_operator_id]

    def execute(self, input: DataFrame, phase='train', stratify=True) -> DataManager:
        # DM caches.
        for node_id in range(len(self.pipeline_operators)):
            self.cached_dm[node_id] = None

        for operator in self.pipeline_operators:
            # Prepare the input dm for the operator.
            if operator.id == 0:
                input_dm = [input]
            else:
                input_dm = [self.cached_dm[id] for id in operator.origins]
            output_dm = operator.operate(input_dm, phase=phase)
            self.cached_dm[operator.id] = output_dm

        final_dm = self.cached_dm[len(self.pipeline_operators) - 1]
        assert isinstance(final_dm, DataManager)
        if phase == 'train':
            final_dm.split(stratify=stratify)
        return final_dm
