from alphaml.engine.components.ensemble.base_ensemble import *
from alphaml.engine.components.data_manager import DataManager
from alphaml.utils.common import get_most
import numpy as np
from functools import reduce


class Bagging(BaseEnsembleModel):
    def __init__(self, model_info, ensemble_size, task_type, metric, model_type='ml'):
        '''
        :param task_type: Majority voting for classification, average for regression
        '''
        super().__init__(model_info, ensemble_size, task_type, metric, model_type)

    def fit(self, dm: DataManager):
        # Train the basic models on this training set.
        if self.model_type == 'ml':
            for config in self.config_list:
                x, y = dm.train_X, dm.train_y
                if dm.val_X is not None:
                    y = list(y)
                    y.extend(list(dm.val_y))
                    y = np.array(y)
                    x = np.vstack((dm.train_X, dm.val_X))
                estimator = self.get_estimator(config, x, y)
                self.ensemble_models.append(estimator)
        elif self.model_type == 'dl':
            pass
        return self

    def predict(self, X):
        # Predict the labels via voting results from the basic models.
        model_pred_list = []
        final_pred = []
        # Get predictions from each model
        from sklearn.metrics import roc_auc_score
        for model in self.ensemble_models:
            pred = self.get_proba_predictions(model, X)
            num_outputs = len(pred)
            model_pred_list.append(pred)

        if self.task_type == CLASSIFICATION:
            if self.metric == roc_auc_score:
                # Calculate the average of predictions
                for i in range(num_outputs):
                    sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
                    pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
                    final_pred.append(pred_average)
            else:
                # Find predictions in majority
                for i in range(num_outputs):
                    sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
                    sample_pred_list = np.array(sample_pred_list)
                    sample_pred_list = list(np.argmax(sample_pred_list, axis=-1))
                    num_majority = get_most(sample_pred_list)
                    final_pred.append(num_majority)

        elif self.task_type == REGRESSION:
            # Calculate the average of predictions
            for i in range(num_outputs):
                sample_pred_list = [model_pred[i] for model_pred in model_pred_list]
                pred_average = reduce(lambda x, y: x + y, sample_pred_list) / len(sample_pred_list)
                final_pred.append(pred_average)

        return np.array(final_pred)
