from alphaml.utils.common import get_max_index

class BaseEnsembleModel(object):
    def __init__(self, model_info, ensemble_size, model_type='ml'):
        self.model_info = model_info
        self.model_type = model_type
        self.ensemble_models = list()
        if len(model_info[0]) < ensemble_size:
            self.ensemble_size = len(model_info[0])
        else:
            self.ensemble_size = ensemble_size

        # Determine the best basic models (with the best performance) from models_infos.
        index_list = get_max_index(self.model_info[1], self.ensemble_size)
        self.config_list = [self.model_info[0][i] for i in index_list]
        for i in index_list:
            print(self.model_info[0][i],self.model_info[1][i])

    def fit(self, dm):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
