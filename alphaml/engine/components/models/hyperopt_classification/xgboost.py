import xgboost as xgb
import numpy as np
from hyperopt import hp

from alphaml.utils.constants import *
from alphaml.engine.components.models.base_model import BaseClassificationModel


class XGBoostClassifier(BaseClassificationModel):
    def __init__(self, n_estimators, eta, min_child_weight, max_depth, subsample, gamma, colsample_bytree,
                 alpha, lambda_t, scale_pos_weight, random_state=None):
        self.n_estimators = n_estimators
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.subsample = subsample
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.alpha = alpha
        self.lambda_t = lambda_t
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = -1
        self.random_state = random_state
        self.num_cls = -1
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        self.n_estimators = int(self.n_estimators)
        dmtrain = xgb.DMatrix(X, label=Y)
        self.num_cls = len(set(Y))

        parameters = dict()
        parameters['eta'] = self.eta
        parameters['min_child_weight'] = self.min_child_weight
        parameters['max_depth'] = self.max_depth
        parameters['subsample'] = self.subsample
        parameters['gamma'] = self.gamma
        parameters['colsample_bytree'] = self.colsample_bytree
        parameters['alpha'] = self.alpha
        parameters['lambda'] = self.lambda_t
        parameters['scale_pos_weight'] = self.scale_pos_weight

        if self.num_cls > 2:
            parameters['num_class'] = self.num_cls
            parameters['objective'] = 'multi:softprob'
            parameters['eval_metric'] = 'merror'
        elif self.num_cls == 2:
            parameters['objective'] = 'binary:logistic'
            parameters['eval_metric'] = 'error'

        parameters['tree_method'] = 'hist'
        parameters['booster'] = 'gbtree'
        parameters['nthread'] = self.n_jobs
        parameters['silent'] = 1
        watchlist = [(dmtrain, 'train')]

        self.estimator = xgb.train(parameters, dmtrain, self.n_estimators, watchlist, verbose_eval=0)
        self.objective = parameters['objective']
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        dm = xgb.DMatrix(X, label=None)
        pred = self.estimator.predict(dm)
        if self.num_cls == 2:
            pred = [int(i > 0.5) for i in pred]
        else:
            pred = np.argmax(pred, axis=-1)
        return np.array(pred)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        dm = xgb.DMatrix(X, label=None)
        pred = self.estimator.predict(dm)
        if self.objective == 'binary:logistic':
            final_pred = np.zeros((pred.shape[0], 2))
            for i, proba in enumerate(pred):
                final_pred[i, :] = np.array([1 - proba, proba])
            return final_pred
        else:
            return self.estimator.predict(dm)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'XGBoost',
                'name': 'XGradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        space = {'n_estimators': hp.randint('xgb_n_estimators', 451) + 50,
                 'eta': hp.loguniform('xgb_eta', np.log(0.025), np.log(0.3)),
                 'min_child_weight': hp.randint('xgb_min_child_weight', 10) + 1,
                 'max_depth': hp.randint('xgb_max_depth', 9) + 2,
                 'subsample': hp.uniform('xgb_subsample', 0.5, 1),
                 'gamma': hp.uniform('xgb_gamma', 0, 1),
                 'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 1),
                 'alpha': hp.loguniform('xgb_alpha', np.log(1e-10), np.log(10)),
                 'lambda_t': hp.loguniform('xgb_lambda_t', np.log(1e-10), np.log(10)),
                 'scale_pos_weight': hp.choice('xgb_scale_pos_weight', [0.01, 0.1, 1, 10, 100])}

        init_trial = {'n_estimators': 200,
                      'eta': 0.3,
                      'min_child_weight': 1,
                      'max_depth': 6,
                      'subsample': 1,
                      'gamma': 0,
                      'colsample_bytree': 1,
                      'alpha': 0,
                      'lambda_t': 1,
                      'scale_pos_weight': 1}

        return space
