from alphaml.engine.components.ensemble.base_ensemble import BaseEnsembleModel
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.evaluator.base import BaseEvaluator
from alphaml.engine.evaluator.dl_evaluator import BaseImgEvaluator
from alphaml.engine.components.models.image_classification import _img_classifiers
from alphaml.utils.common import get_most
import numpy as np
from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, model_info, ensemble_size, model_type='ml', bagging_mode='majority'):
        '''
        :param bagging_mode: string, mode for bagging, 'majority' and 'average'
        '''
        super().__init__(model_info, ensemble_size, model_type)
        self.bagging_mode = bagging_mode


    def fit(self, dm: DataManager):
        # Train the basic models on this training set.
        if self.model_type == 'ml':
            for config in self.config_list:
                evaluator = BaseEvaluator()
                _, estimator = evaluator.set_config(config)
                estimator.fit(dm.train_X, dm.train_y)
                self.ensemble_models.append(estimator)
        elif self.model_type == 'dl':
            pass
        return self

    def predict(self, X):
        # Predict the labels via voting results from the basic models.
        model_pred_list = []
        final_pred = []
        if self.bagging_mode == 'majority':
            # Get predictions from each model
            for model in self.ensemble_models:
                pred = model.predict(X)
                num_outputs = len(pred)
                model_pred_list.append(pred)
            # Find predictions in majority
            for i in range(num_outputs):
                sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
                num_majority = get_most(sample_pred_list)
                final_pred.append(num_majority)

        # elif self.bagging_mode == 'average':
        #     for model in self.ensemble_models:
        #         pred = model.predict_proba(X)
        #         num_outputs = len(pred)
        #         model_pred_list.append(pred)
        #     for i in range(num_outputs):
        #         sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
        #         proba_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
        #         pred_average = np.argmax(proba_average, axis=-1)
        #         final_pred.append(pred_average)

        return np.array(final_pred)
