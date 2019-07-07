import numpy as np
import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from time import time


class TreeNode:

    def __init__(self, feature_key):
        self.feature_key = feature_key
        self.performance = None

    def set_performance(self, model, train_data, train_label, valid_data, valid_label):

        model.fit(train_data, train_label)
        y_pred = model.predict(valid_data)
        self.performance = accuracy_score(valid_label, y_pred)

    def __lt__(self, other):
        return self.performance < other.performance

    def __le__(self, other):
        return self.performance <= other.performance

    def __gt__(self, other):
        return self.performance > other.performance

    def __ge__(self, other):
        return self.performance >= other.performance


"""
AutoCross is a feature generated algorithm based on the KDD 2019 paper.
"""


class AutoCross:

    def __init__(self, max_iter, model=LogisticRegression(), metrics="accuracy"):
        self.max_iter = max_iter
        self.model = model
        self.metrics = metrics

        self.feature_sets = None
        self.feature_cols = dict()
        self.train_data = None
        self.valid_data = None

    def _get_node(self):
        pass

    def _greedy_search(self, x_train, x_valid, y_train, y_valid):
        feature_num = x_train.shape[1]
        self.feature_sets = [[feature_id] for feature_id in range(feature_num)]
        for feature_id in range(feature_num):
            self.feature_cols[str(feature_id)] = dict()
            self.feature_cols[str(feature_id)]["train"] = x_train[:, feature_id]
            self.feature_cols[str(feature_id)]["valid"] = x_valid[:, feature_id]

        for iteration in range(self.max_iter):
            # generate new features via cross operation
            best_feature = None
            best_performance = 0
            for i in range(len(self.feature_sets)):
                for j in range(i + 1, len(self.feature_sets)):
                    set1 = self.feature_sets[i]
                    set2 = self.feature_sets[j]
                    cross_set = set1 + set2
                    feature_key = ','.join(str(feature) for feature in cross_set)

                    if self.feature_cols.get(feature_key) is None:
                        set1_key = ','.join(str(feature) for feature in set1)
                        set2_key = ','.join(str(feature) for feature in set2)

                        self.feature_cols[feature_key] = dict()
                        self.feature_cols[feature_key]["train"] = \
                            self.feature_cols[set1_key]["train"] * self.feature_cols[set2_key]["train"]
                        self.feature_cols[feature_key]["valid"] = \
                            self.feature_cols[set1_key]["valid"] * self.feature_cols[set2_key]["valid"]

                    if self.train_data is None:
                        train_data = self.feature_cols[feature_key]["train"].reshape((-1, 1))
                    else:
                        train_data = \
                            np.hstack((self.train_data, self.feature_cols[feature_key]["train"].reshape((-1, 1))))

                    if self.valid_data is None:
                        valid_data = self.feature_cols[feature_key]["valid"].reshape((-1, 1))
                    else:
                        valid_data = \
                            np.hstack((self.valid_data, self.feature_cols[feature_key]["valid"].reshape((-1, 1))))

                    self.model.fit(train_data, y_train)
                    y_pred = self.model.predict(valid_data)
                    perf = accuracy_score(y_valid, y_pred)

                    if perf > best_performance:
                        best_feature = feature_key
                        best_performance = perf

            best_feature_list = [int(feature) for feature in best_feature.split(',')]
            self.feature_sets.append(best_feature_list)
            if self.train_data is None:
                self.train_data = self.feature_cols[best_feature]["train"].reshape(-1, 1)
            else:
                self.train_data = \
                    np.hstack((self.train_data, self.feature_cols[best_feature]["train"].reshape(-1, 1)))

            if self.valid_data is None:
                self.valid_data = self.feature_cols[best_feature]["valid"].reshape(-1, 1)
            else:
                self.valid_data = \
                    np.hstack((self.valid_data, self.feature_cols[best_feature]["valid"].reshape(-1, 1)))

            print("iteration:", iteration, "performance:", best_performance)

    def _successive_halving(self, x_train, x_valid, y_train, y_valid):
        feature_num = x_train.shape[1]
        self.feature_sets = [[feature_id] for feature_id in range(feature_num)]
        for feature_id in range(feature_num):
            self.feature_cols[str(feature_id)] = dict()
            self.feature_cols[str(feature_id)]["train"] = x_train[:, feature_id]
            self.feature_cols[str(feature_id)]["valid"] = x_valid[:, feature_id]

        for iteration in range(self.max_iter):
            t0 = time()
            nodes = list()
            for i in range(len(self.feature_sets)):
                for j in range(i + 1, len(self.feature_sets)):
                    set1 = self.feature_sets[i]
                    set2 = self.feature_sets[j]
                    cross_set = set1 + set2
                    feature_key = ','.join(str(feature) for feature in cross_set)

                    if self.feature_cols.get(feature_key) is None:
                        set1_key = ','.join(str(feature) for feature in set1)
                        set2_key = ','.join(str(feature) for feature in set2)

                        # print(self.feature_cols[set1_key]["train"].shape)
                        # print(self.feature_cols[set2_key]["train"].shape)
                        self.feature_cols[feature_key] = dict()
                        self.feature_cols[feature_key]["train"] = \
                            self.feature_cols[set1_key]["train"] * self.feature_cols[set2_key]["train"]
                        self.feature_cols[feature_key]["valid"] = \
                            self.feature_cols[set1_key]["valid"] * self.feature_cols[set2_key]["valid"]

                    node = TreeNode(feature_key=feature_key)
                    nodes.append(node)

            # start the successive halving process
            n = len(nodes)  # n is the number of the configurations / bandits
            B = len(x_train)
            r = int(B / n)
            if r < 50:
                r = 100
            while n > 1 and r <= B:
                print("n =", n, "r =", r)
                n_samples = np.random.choice(B, r, replace=False)
                for node in nodes:
                    if self.train_data is None:
                        train_data = self.feature_cols[node.feature_key]["train"].reshape((-1, 1))[n_samples]
                    else:
                        train_data = \
                            np.hstack((self.train_data,
                                       self.feature_cols[node.feature_key]["train"].reshape((-1, 1))))[n_samples]

                    if self.valid_data is None:
                        valid_data = self.feature_cols[node.feature_key]["valid"].reshape((-1, 1))
                    else:
                        valid_data = \
                            np.hstack((self.valid_data,
                                       self.feature_cols[node.feature_key]["valid"].reshape((-1, 1))))

                    node.set_performance(model=self.model, train_data=train_data, train_label=y_train[n_samples],
                                         valid_data=valid_data, valid_label=y_valid)
                nodes.sort(reverse=True)
                if int(n / 2) >= 1:
                    nodes = nodes[:int(n / 2)]
                n /= 2
                r *= 2

            best_node = nodes[0]
            best_feature_list = [int(feature) for feature in best_node.feature_key.split(',')]
            self.feature_sets.append(best_feature_list)
            if self.train_data is None:
                self.train_data = self.feature_cols[best_node.feature_key]["train"].reshape(-1, 1)
            else:
                self.train_data = \
                    np.hstack((self.train_data, self.feature_cols[best_node.feature_key]["train"].reshape(-1, 1)))

            if self.valid_data is None:
                self.valid_data = self.feature_cols[best_node.feature_key]["valid"].reshape(-1, 1)
            else:
                self.valid_data = \
                    np.hstack((self.valid_data, self.feature_cols[best_node.feature_key]["valid"].reshape(-1, 1)))

            self.model.fit(np.hstack((x_train, self.train_data)), y_train)
            y_pred = self.model.predict(np.hstack((x_valid, self.valid_data)))
            perf = accuracy_score(y_valid, y_pred)
            # if iteration % 5 == 0:
            print("iteration:", iteration, " performance:", best_node.performance, "all data perf:", perf," time:", time() - t0)

    def _hyperband(self, x_train, x_valid, y_train, y_valid, eta=3):
        feature_num = x_train.shape[1]
        self.feature_sets = [[feature_id] for feature_id in range(feature_num)]
        for feature_id in range(feature_num):
            self.feature_cols[str(feature_id)] = dict()
            self.feature_cols[str(feature_id)]["train"] = x_train[:, feature_id]
            self.feature_cols[str(feature_id)]["valid"] = x_valid[:, feature_id]

        best_node = None
        for iteration in range(self.max_iter):
            t0 = time()
            # start the hyperband process
            R = len(x_train)
            s_max = int(math.log(R, eta))
            B = (s_max + 1) * R
            for s in range(s_max, -1, -1):
                nodes = []
                subset_records = set()  # record the subset has
                n = math.ceil((B / R) * (pow(eta, s) / (s + 1)))
                r = int(R * pow(eta, -s)) * int(R / 100)
                if r > R:
                    break
                if n > len(self.feature_sets) * (len(self.feature_sets) - 1) // 2:
                    # n = len(self.feature_sets) * (len(self.feature_sets) - 1) // 2
                    n = int((len(self.feature_sets) * (len(self.feature_sets) - 1) // 2) / pow(3, s_max - s))
                    if n <= 1:
                        break
                    for i in range(len(self.feature_sets)):
                        for j in range(i + 1, len(self.feature_sets)):
                            set1 = self.feature_sets[i]
                            set2 = self.feature_sets[j]
                            cross_set = set1 + set2
                            feature_key = ','.join(str(feature) for feature in cross_set)

                            if self.feature_cols.get(feature_key) is None:
                                set1_key = ','.join(str(feature) for feature in set1)
                                set2_key = ','.join(str(feature) for feature in set2)

                                self.feature_cols[feature_key] = dict()
                                self.feature_cols[feature_key]["train"] = \
                                    self.feature_cols[set1_key]["train"] * self.feature_cols[set2_key]["train"]
                                self.feature_cols[feature_key]["valid"] = \
                                    self.feature_cols[set1_key]["valid"] * self.feature_cols[set2_key]["valid"]

                            node = TreeNode(feature_key=feature_key)
                            nodes.append(node)
                else:

                    while len(nodes) < n:
                        p1, p2 = np.random.choice(len(self.feature_sets), 2, replace=False)
                        set1 = self.feature_sets[p1]
                        set2 = self.feature_sets[p2]
                        cross_set = set1 + set2
                        feature_key = ','.join(str(feature) for feature in cross_set)
                        if feature_key in subset_records:
                            continue
                        subset_records.add(feature_key)

                        if self.feature_cols.get(feature_key) is None:
                            set1_key = ','.join(str(feature) for feature in set1)
                            set2_key = ','.join(str(feature) for feature in set2)

                            self.feature_cols[feature_key] = dict()
                            self.feature_cols[feature_key]["train"] = \
                                self.feature_cols[set1_key]["train"] * self.feature_cols[set2_key]["train"]
                            self.feature_cols[feature_key]["valid"] = \
                                self.feature_cols[set1_key]["valid"] * self.feature_cols[set2_key]["valid"]

                        node = TreeNode(feature_key=feature_key)
                        nodes.append(node)
                # start the inner loop
                while n > 1 and r <= R:
                    print("n=", n, "r=", r)
                    n_samples = np.random.choice(R, r, replace=False)
                    for node in nodes:
                        if self.train_data is None:
                            train_data = self.feature_cols[node.feature_key]["train"].reshape((-1, 1))[n_samples]
                        else:
                            train_data = \
                                np.hstack((self.train_data,
                                           self.feature_cols[node.feature_key]["train"].reshape((-1, 1))))[n_samples]

                        if self.valid_data is None:
                            valid_data = self.feature_cols[node.feature_key]["valid"].reshape((-1, 1))
                        else:
                            valid_data = \
                                np.hstack((self.valid_data,
                                           self.feature_cols[node.feature_key]["valid"].reshape((-1, 1))))

                        node.set_performance(model=self.model, train_data=train_data, train_label=y_train[n_samples],
                                             valid_data=valid_data, valid_label=y_valid)
                    nodes.sort(reverse=True)
                    n = math.ceil(n / eta)
                    r = int(r * eta)
                    if n >= 1:
                        nodes = nodes[:n]

                if best_node is None:
                    best_node = nodes[0]
                else:
                    if nodes[0].performance > best_node.performance:
                        best_node = nodes[0]

                print("outer loop:", iteration, "inner loop:", s, "performance:",
                      best_node.performance, " time:",
                      time() - t0, flush=True)

            best_feature_list = [int(feature) for feature in best_node.feature_key.split(',')]
            self.feature_sets.append(best_feature_list)
            if self.train_data is None:
                self.train_data = self.feature_cols[best_node.feature_key]["train"].reshape(-1, 1)
            else:
                self.train_data = \
                    np.hstack((self.train_data, self.feature_cols[best_node.feature_key]["train"].reshape(-1, 1)))

            if self.valid_data is None:
                self.valid_data = self.feature_cols[best_node.feature_key]["valid"].reshape(-1, 1)
            else:
                self.valid_data = \
                    np.hstack((self.valid_data, self.feature_cols[best_node.feature_key]["valid"].reshape(-1, 1)))

    def fit(self, x_train, x_valid, y_train, y_valid):
        # self._greedy_search(x_train, x_valid, y_train, y_valid)
        # self._successive_halving(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)
        self._hyperband(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)

    def transform(self):
        # generated_features = self
        return self.train_data, self.valid_data
