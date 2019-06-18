from alphaml.engine.components.ensemble.base_ensemble import BaseEnsembleModel
from alphaml.engine.components.data_manager import DataManager
from alphaml.engine.evaluator.base import BaseEvaluator
from alphaml.engine.evaluator.dl_evaluator import BaseImgEvaluator
from alphaml.engine.components.models.image_classification import _img_classifiers
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


class Stacking(BaseEnsembleModel):
    def __init__(self, model_info, ensemble_size, model_type='ml', stacking_model='xgboost', kfold=5):
        '''
        :param bagging_mode: string, mode for bagging, 'majority' and 'average'
        '''
        super().__init__(model_info, ensemble_size, model_type)
        self.kfold = kfold
        # We use LogisticRegressor as default blending model
        if stacking_model == 'logistic':
            from sklearn.linear_model.logistic import LogisticRegression
            self.stacking_model = LogisticRegression(max_iter=1000)
        elif stacking_model == 'gb':
            from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
            self.stacking_model = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                             n_estimators=200)
        elif stacking_model == 'xgboost':
            from xgboost import XGBClassifier
            self.stacking_model = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200)

    def fit(self, dm: DataManager):
        # Split training data for phase 1 and phase 2
        skf = StratifiedKFold(n_splits=self.kfold)
        feature_p2 = None
        if self.model_type == 'ml':
            # Train basic models using a part of training data
            for i, config in enumerate(self.config_list):
                for j, (train, test) in enumerate(skf.split(dm.train_X, dm.train_y)):
                    evaluator = BaseEvaluator()
                    clf_type, estimator = evaluator.set_config(config)
                    x_p1, x_p2, y_p1, _ = dm.train_X[train], dm.train_X[test], dm.train_y[train], dm.train_y[test]
                    estimator.fit(x_p1, y_p1)
                    self.ensemble_models.append(
                        estimator)  # The final list will contain self.kfold * self.ensemble_size models
                    pred = estimator.predict_proba(x_p2)
                    n_dim = np.array(pred).shape[1]
                    if n_dim == 2:
                        # Binary classificaion
                        n_dim = 1
                    # Initialize training matrix for phase 2
                    if feature_p2 is None:
                        num_samples = len(train) + len(test)
                        feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                    if n_dim == 1:
                        feature_p2[test, i * n_dim:(i + 1) * n_dim] = pred[:, 1:2]
                    else:
                        feature_p2[test, i * n_dim:(i + 1) * n_dim] = pred
            # Train model for stacking using the other part training data
            self.stacking_model.fit(feature_p2, dm.train_y)

            from sklearn.metrics import accuracy_score
            pred = self.stacking_model.predict(feature_p2)
            print(accuracy_score(dm.train_y, pred))

        elif self.model_type == 'dl':
            pass
        return self

    def predict(self, X):
        # Predict the labels via stacking
        feature_p2 = None
        for i, model in enumerate(self.ensemble_models):
            pred = model.predict_proba(X)
            n_dim = np.array(pred).shape[1]
            if n_dim == 2:
                n_dim = 1
            if feature_p2 is None:
                num_samples = np.array(X).shape[0]
                feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
            index = i % self.kfold

            # Get average predictions
            if n_dim == 1:
                feature_p2[:, index * n_dim:(index + 1) * n_dim] = feature_p2[:, index * n_dim:(index + 1) * n_dim] + \
                                                                   pred[:, 1:2] / self.kfold
            else:
                feature_p2[:, index * n_dim:(index + 1) * n_dim] = feature_p2[:, index * n_dim:(index + 1) * n_dim] + \
                                                                   pred / self.kfold
        # Get predictions from blending model
        final_pred = self.stacking_model.predict(feature_p2)
        return final_pred
