import pandas as pd
import numpy as np

class MyKNNReg():

    def __init__(self, k=3, metric="euclidean", weight="uniform"):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        components = self.__dict__
        class_name = self.__class__.__name__
        components_str = ", ".join(f"{key}={value}" for key, value in components.items())
        return f"{class_name} class: {components_str}"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.train_size = X_train.shape

    def _euclidean_distance(self, row: pd.Series):
        distance = (row - self.X_train).pow(2).sum(axis=1).pow(.5)
        return distance

    def _chebyshev_distance(self, row: pd.Series):
        distance = (row - self.X_train).abs().max(axis=1)
        return distance

    def _manhattan_distance(self, row: pd.Series):
        distance = (row - self.X_train).abs().sum(axis=1)
        return distance

    def _cosine_distance(self, row: pd.Series):
        cosine_similarity = np.dot(self.X_train, row) / (np.linalg.norm(self.X_train, axis=1) * (np.linalg.norm(row)))
        distance = 1 - cosine_similarity
        return pd.Series(distance, index=self.X_train.index)

    def _get_weights(self, distances):
        if self.weight == "uniform":
            return np.ones_like(distances)
        elif self.weight == "rank":
            ranks = np.argsort(np.argsort(distances))
            return 1 / (ranks + 1)
        elif self.weight == "distance":
            return 1 / distances


    def _predict(self, row: pd.Series):
        if self.metric == "euclidean":
            distances = self._euclidean_distance(row)
        elif self.metric == "chebyshev":
            distances = self._chebyshev_distance(row)
        elif self.metric == "manhattan":
            distances = self._manhattan_distance(row)
        elif self.metric == "cosine":
            distances = self._cosine_distance(row)
        else:
            raise ValueError("Unsupported metric: {}".format(self.metric))

        k_sorted_idx = distances.sort_values().head(self.k).index
        nearest_distances = distances[k_sorted_idx]
        nearest_labels = self.y_train.iloc[k_sorted_idx]
        if self.weight == "uniform":
            y_pred = nearest_labels.mean()
        elif self.weight == "rank" or self.weight == "distance":
            weights = self._get_weights(nearest_distances)
            y_pred = (weights * nearest_labels).sum() / weights.sum()


        return y_pred



    def predict(self, X_test: pd.DataFrame):
        result = X_test.apply(self._predict, axis=1)
        return result



from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

model = MyKNNReg(k=5, metric="cosine", weight="distance")
model.fit(X, y)

print(model)

X_test, y_test = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5, random_state=42)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]

print(f"result of prediction: {model.predict(X_test).sum()}")