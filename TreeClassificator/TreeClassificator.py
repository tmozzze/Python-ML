import numpy as np
import pandas as pd

class MyTreeClf():

    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self):
        components = self.__dict__
        class_name = self.__class__.__name__
        components_str = ", ".join(f"{key}={value}" for key, value in components.items())
        return f"{class_name} class: {components_str}"

    def entropy(self, y):
        

    def information_gain(self, y, y_left, y_right):
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        pearent_entropy = self.entropy(y)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        for col in X.columns:
            unique_values = sorted(X[col].unique())

            for i in range (1, len(unique_values)):
                split_value = unique_values[i]

                left_mask = X[col] <= split_value
                right_mask = X[col] > split_value

                y_left, y_right = y[left_mask], y[right_mask]

                ig = self.information_gain(y, y_left, y_right)




from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

model = MyTreeClf()
print(model)