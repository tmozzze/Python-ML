import numpy as np
import pandas as pd


class MyKNNClf():
    def __init__(self, k=3):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None

    def __str__(self):
        components = self.__dict__
        class_name = self.__class__.__name__
        components_str = ", ".join(f"{key}={value}" for key, value in components.items())
        return f"{class_name} class: {components_str}"

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def predict(self, X_test):
        



from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

model = MyKNNClf(k=20)
print(model)
model.fit(X, y)
print(model)