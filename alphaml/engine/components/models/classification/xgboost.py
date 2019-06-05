import xgboost as xgb
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
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
            parameters['objective'] = 'multi:softmax'
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
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        dm = xgb.DMatrix(X, label=None)
        pred = self.estimator.predict(dm)
        if self.num_cls == 2:
            pred = [int(i > 0.5) for i in pred]
        return pred

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        dm = xgb.DMatrix(X, label=None)
        return self.estimator.predict_proba(dm)

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
        cs = ConfigurationSpace()

        n_estimators = UniformFloatHyperparameter("n_estimators", 50, 500, default_value=200, q=20)
        eta = UniformFloatHyperparameter("eta", 0.025, 0.3, default_value=0.3, q=0.025)
        min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
        max_depth = UniformIntegerHyperparameter("max_depth", 2, 10, default_value=6)
        subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1, q=0.05)
        gamma = UniformFloatHyperparameter("gamma", 0, 1, default_value=0, q=0.1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1., q=0.05)
        alpha = UniformFloatHyperparameter("alpha", 0, 10, default_value=0., q=1.)
        lambda_t = UniformFloatHyperparameter("lambda_t", 1, 2, default_value=1, q=0.1)
        scale_pos_weight = CategoricalHyperparameter("scale_pos_weight", [0.01, 0.1, 1., 10, 100], default_value=1.)

        cs.add_hyperparameters(
            [n_estimators, eta, min_child_weight, max_depth, subsample, gamma, colsample_bytree, alpha, lambda_t,
             scale_pos_weight])
        return cs
