from alphaml.engine.components.models.image_classification import _img_classifiers
from alphaml.engine.evaluator.base import BaseEvaluator, update_config


class BaseImgEvaluator(BaseEvaluator):
    def __init__(self, inputshape, classnum):
        super().__init__()
        self.inputshape = inputshape
        self.classnum = classnum

    def __call__(self, config):
        params_num = len(config.get_dictionary().keys()) - 1
        classifier_type = config['estimator']
        estimator = _img_classifiers[classifier_type]()
        config = update_config(config)
        estimator.set_hyperparameters(config)
        estimator.set_model_config(self.inputshape, self.classnum)
        # Fit the estimator on the training data.
        kwargs = {}
        kwargs['metric'] = self.metric_func
        estimator.fit(self.data_manager, **kwargs)

        # Get the best result on val data
        metric = estimator.best_result

        # Turn it to a minimization problem.
        return 1 - metric

    def predict(self, config, test_X=None, **kwargs):
        if not hasattr(self, 'estimator'):
            # Build the corresponding estimator.
            params_num = len(config.get_dictionary().keys()) - 1
            classifier_type = config['estimator']
            estimator = _img_classifiers[classifier_type](*[None] * params_num)
        else:
            estimator = self.estimator
        config = update_config(config)
        estimator.set_hyperparameters(config)

        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y)

        # Inference.
        if test_X is None:
            test_X = self.data_manager.test_X
        y_pred = estimator.predict(test_X)
        return y_pred
