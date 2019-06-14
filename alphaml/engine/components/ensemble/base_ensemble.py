class BaseEnsembleModel(object):
    def __init__(self, model_info, ensemble_size, model_type='ml'):
        self.model_info = model_info
        self.model_type = model_type
        self.ensemble_models = list()
        if len(model_info) < ensemble_size:
            self.ensemble_size = len(model_info)
        else:
            self.ensemble_size = ensemble_size

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
