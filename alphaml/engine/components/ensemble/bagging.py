from alphaml.engine.components.ensemble.base_ensemble import BaseEnsembleModel


class Bagging(BaseEnsembleModel):
    def __init__(self,  model_infos, ensemble_size):
        super().__init__(model_infos, ensemble_size)

        # Determine best basic models from models_infos.

    def fit(self, dm):
        # Train the basic models on this training set.
        pass

    def predict(self, X):
        # predict the labels via voting results from the basic models.
        pass
