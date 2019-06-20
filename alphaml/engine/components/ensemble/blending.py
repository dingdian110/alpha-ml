from alphaml.engine.components.ensemble.base_ensemble import BaseEnsembleModel
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.evaluator.base import BaseEvaluator
from alphaml.engine.evaluator.dl_evaluator import BaseImgEvaluator
from alphaml.engine.components.models.image_classification import _img_classifiers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier


class Blending(BaseEnsembleModel):
    def __init__(self, model_info, ensemble_size, model_type='ml', meta_learner='xgboost'):
        '''
        :param bagging_mode: string, mode for bagging, 'majority' and 'average'
        '''
        super().__init__(model_info, ensemble_size, model_type)

        # We use LogisticRegressor as default blending model
        if meta_learner == 'logistic':
            self.meta_learner = LogisticRegression(max_iter=1000)
        elif meta_learner == 'gb':
            self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                           n_estimators=250)
        elif meta_learner == 'xgboost':
            from xgboost import XGBClassifier
            self.meta_learner = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200)

    def fit(self, dm: DataManager):
        # Split training data for phase 1 and phase 2
        x_p1, x_p2, y_p1, y_p2 = train_test_split(dm.train_X, dm.train_y, test_size=0.2, stratify=dm.train_y)
        feature_p2 = None
        if self.model_type == 'ml':
            # Train basic models using a part of training data
            for i, config in enumerate(self.config_list):
                evaluator = BaseEvaluator()
                clf_type, estimator = evaluator.set_config(config)
                estimator.fit(x_p1, y_p1)
                self.ensemble_models.append(estimator)
                pred = estimator.predict_proba(x_p2)
                n_dim = np.array(pred).shape[1]
                if n_dim == 2:
                    # Binary classificaion
                    n_dim = 1
                # Initialize training matrix for phase 2
                if feature_p2 is None:
                    num_samples = np.array(x_p2).shape[0]
                    feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                if n_dim == 1:
                    feature_p2[:, i * n_dim:(i + 1) * n_dim] = pred[:, 1:2]
                else:
                    feature_p2[:, i * n_dim:(i + 1) * n_dim] = pred
            # Train model for blending using the other part of training data
            self.meta_learner.fit(feature_p2, y_p2)


        elif self.model_type == 'dl':
            pass
        return self

    def predict(self, X):
        # Predict the labels via blending
        feature_p2 = None
        for i, model in enumerate(self.ensemble_models):
            pred = model.predict_proba(X)
            n_dim = np.array(pred).shape[1]
            if n_dim == 2:
                n_dim = 1
            if feature_p2 is None:
                num_samples = np.array(X).shape[0]
                feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
            if n_dim == 1:
                feature_p2[:, i * n_dim:(i + 1) * n_dim] = pred[:, 1:2]
            else:
                feature_p2[:, i * n_dim:(i + 1) * n_dim] = pred
        # Get predictions from meta-learner
        final_pred = self.meta_learner.predict(feature_p2)
        return final_pred
