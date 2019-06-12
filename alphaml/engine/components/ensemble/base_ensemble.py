class BaseEnsembleModel(object):
    def __init__(self, model_infos, ensemble_size):
        self.model_infos = model_infos
        self.ensemble_size = ensemble_size
        self.basic_models = list()

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
