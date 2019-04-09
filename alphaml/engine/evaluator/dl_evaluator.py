from alphaml.engine.components.models.image_classification import _img_classifiers
from alphaml.engine.evaluator.base import BaseEvaluator, update_config


class DLEvaluator(BaseEvaluator):
    def __call__(self, config):
        params_num = len(config.get_dictionary().keys()) - 1
        classifier_type = config['estimator']
        estimator = _img_classifiers[classifier_type](*[None]*params_num)
        config = update_config(config)
        estimator.set_hyperparameters(config)

        # Fit the estimator on the training data.
        estimator.fit(self.data_manager.train_X, self.data_manager.train_y,
                      self.data_manager.val_X,self.data_manager.val_y)

        # Validate it on val data.
        y_pred = estimator.predict(self.data_manager.val_X)
        metric = self.metric_func(self.data_manager.val_y, y_pred)

        # Turn it to a minimization problem.
        return 1 - metric
