import numpy as np

from alphaml.engine.components.feature_engineering.auto_cross import AutoCross
from alphaml.engine.components.feature_engineering.selector import RandomForestSelector

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class AutoFeature:

    def __init__(self, strategy="auto_cross"):
        self.strategy = strategy
        self.solver = None

    def fit(self, dm, generated_num=50):
        if self.strategy == "auto_cross":
            self.solver = AutoCross(generated_num)
        elif self.strategy == "auto_deep":
            self.solver = None
        elif self.strategy == "auto_learn":
            self.solver = None
        else:
            raise ValueError("AutoFeature supports only strategies in ['autocross', 'auto_deep', 'auto_learn', "
                             "got", self.strategy, ".")
        self.solver.fit(dm.train_X, dm.val_X, dm.train_y, dm.val_y)

    def transform(self, dm):
        generated_train_data, generated_valid_data = self.solver.tranform()
        feature_num = dm.train_X.shape[1]

        if feature_num < 20:
            dm.train_X = generated_train_data
            dm.val_X = generated_valid_data
        else:
            print("start selection process...............")
            selector = RandomForestSelector()
            selector.fit(dm.train_X, dm.train_y)

            lr = LogisticRegression()
            lr.fit(generated_train_data, dm.train_y)
            y_pred = lr.predict(generated_valid_data)
            best_perf = accuracy_score(dm.train_y, y_pred)
            best_k = 0
            for percentile in range(1, 10):
                k = int((percentile / 10.0) * feature_num)
                selected_train_data = selector.transform(dm.train_X, k)
                selected_valid_data = selector.transform(dm.val_X, k)
                lr.fit(np.hstack((generated_train_data, selected_train_data)), dm.train_y)
                y_pred = lr.predict(np.hstack((generated_valid_data, selected_valid_data)))
                perf = accuracy_score(dm.valid_y, y_pred)
                if perf <= best_perf:
                    break
                else:
                    print("selected pertentile:", percentile, "perf:", best_perf, flush=True)
                    best_perf = perf
                    best_k = k
            if best_k != 0:
                dm.train_X = np.hstack((generated_train_data, selector.transform(dm.train_X, best_k)))
                dm.val_X = np.hstack((generated_valid_data, selector.transform(dm.val_X, best_k)))
            else:
                dm.train_X = generated_train_data
                dm.val_X = generated_valid_data

        return dm
