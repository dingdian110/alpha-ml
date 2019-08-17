import numpy as np
from hyperopt import hp

from alphaml.utils.constants import *
from alphaml.utils.model_util import softmax
from alphaml.utils.common import check_none, check_for_bool
from alphaml.engine.components.models.base_model import BaseClassificationModel


class LibSVM_SVC(BaseClassificationModel):
    def __init__(self, C, kernel, gamma, shrinking, tol, max_iter,
                 class_weight=None, degree=3, coef0=0, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        # Nested kernel
        nested_kernel = self.kernel
        self.kernel = nested_kernel[0]
        if self.kernel == 'poly':
            self.degree = nested_kernel[1]['degree']
            self.coef0 = nested_kernel[1]['coef0']
        elif self.kernel == 'sigmoid':
            self.coef0 = nested_kernel[1]['coef0']
        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(C=self.C,
                                         kernel=self.kernel,
                                         degree=self.degree,
                                         gamma=self.gamma,
                                         coef0=self.coef0,
                                         shrinking=self.shrinking,
                                         tol=self.tol,
                                         class_weight=self.class_weight,
                                         max_iter=self.max_iter,
                                         random_state=self.random_state,
                                         decision_function_shape='ovr')
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LibSVM-SVC',
                'name': 'LibSVM Support Vector Classification',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, SPARSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        coef0 = hp.uniform("libsvm_coef0", -1, 1)
        space = {'C': hp.loguniform('libsvm_C', np.log(0.03125), np.log(32768)),
                 'gamma': hp.loguniform('libsvm_gamma', np.log(3.0517578125e-5), np.log(8)),
                 'shrinking': hp.choice('libsvm_shrinking', ["True", "False"]),
                 'tol': hp.loguniform('libsvm_tol', np.log(1e-5), np.log(1e-1)),
                 'max_iter': hp.choice('libsvm_max_iter', [2000]),
                 'kernel': hp.choice('libsvm_kernel',
                                     [("poly", {'degree': hp.randint('libsvm_degree', 4) + 2, 'coef0': coef0}),
                                      ("rbf", {}),
                                      ("sigmoid", {'coef0': coef0})])}

        init_trial = {'C': 1,
                      'gamma': 0.1,
                      'shrinking': "True",
                      'tol': 1e-3,
                      'max_iter': 2000,
                      'kernel': ("rbf", {})}

        return space
