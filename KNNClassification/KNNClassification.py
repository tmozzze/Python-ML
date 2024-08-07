import numpy as np
import pandas as pd


class MyKNNClf():
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric

    def __str__(self):
        components = self.__dict__
        class_name = self.__class__.__name__
        components_str = ", ".join(f"{key}={value}" for key, value in components.items())
        return f"{class_name} class: {components_str}"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.train_size = X.shape

    def _euclidean_proba(self, row: pd.Series):
        k_sorted_idx = (row - self.X_train).pow(2).sum(axis=1).pow(.5).sort_values().head(self.k).index
        return self.y_train[k_sorted_idx].mean()

    def _chebyshev_proba(self, row: pd.Series):
        k_sorted_idx = (row - self.X_train).abs().max(axis=1).sort_values().head(self.k).index
        return self.y_train[k_sorted_idx].mean()

    def _manhattan_proba(self, row: pd.Series):
        k_sorted_idx = (row - self.X_train).abs().sum(axis=1).sort_values().head(self.k).index
        return self.y_train[k_sorted_idx].mean()

    def _cosine_proba(self, row: pd.Series):
        k_sorted_idx = np.argsort(1 - (np.dot(self.X_train, row) / (np.linalg.norm(self.X_train, axis=1) * (np.linalg.norm(row)))))[:self.k]
        return self.y_train.iloc[k_sorted_idx].mean()

    def _predict_proba(self, row: pd.Series):
        if self.metric == "euclidean":
            return self._euclidean_proba(row)
        if self.metric == "chebyshev":
            return self._chebyshev_proba(row)
        if self.metric == "manhattan":
            return self._manhattan_proba(row)
        if self.metric == "cosine":
            return self._cosine_proba(row)
        else:
            raise ValueError("Unsupported metric: {}".format(self.metric))

    def _predict(self, row: pd.Series):
        mean = self._predict_proba(row)
        return 1 if mean >= 0.5 else 0

    def predict_proba(self, X_test: pd.DataFrame):
        probabilities = X_test.apply(self._predict_proba, axis=1)
        return probabilities

    def predict(self, X_test: pd.DataFrame):
        predictions = X_test.apply(self._predict, axis=1)
        return predictions





from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

model = MyKNNClf(k=5, metric="cosine")
print(model)
model.fit(X, y)
print(model)

X_test, y_test = make_classification(n_samples=400, n_features=14, n_informative=10, random_state=42)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]

print(f"Test predict proba: {sum(model.predict_proba(X_test))}")
print(f"Test predict: {sum(model.predict(X_test))}")