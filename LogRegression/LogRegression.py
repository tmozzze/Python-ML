import numpy
import numpy as np
import pandas as pd

class MyLogReg():

    def __init__(self, n_iter=10, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights


    def __str__(self):
        components = self.__dict__
        class_name = self.__class__.__name__
        components_str = ", ".join(f"{key}={value}" for key, value in components.items())
        return f"{class_name} class: {components_str}"

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def compute_log_loss(self, y, y_hat):
        eps = 1e-15
        log_loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        return log_loss

    def fit(self, X, y, verbose=False):
        X = np.c_[np.ones(X.shape[0]), X]

        self.weights = np.ones(X.shape[1])



        for i in range(self.n_iter):
            y_hat = self.sigmoid(np.dot(X, self.weights))

            loss = self.compute_log_loss(y, y_hat)

            if verbose and (i % verbose == 0):
                print(f"{i if i > 0 else 'start'} | loss: {loss:.2f} | learning rate: {self.learning_rate}")

            error = y_hat - y
            gradient = np.dot(error, X) / len(y)

            self.weights -= self.learning_rate * gradient

        if verbose:
            print(f"final | loss: {loss:.2f} | learning rate: {self.learning_rate}")

    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        y_hat = self.sigmoid(np.dot(X, self.weights))
        return y_hat

    def predict(self, X):
        y_proba = self.predict_proba(X)
        result = np.where(y_proba > 0.5, 1, 0)
        return result

    def get_coef(self):
        return self.weights[1:]


from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]



model = MyLogReg(n_iter=100, learning_rate=0.1)

print(model)

model.fit(X, y, 10)

print(model.get_coef())

X_test, y_test = make_classification(n_samples=400, n_features=14, n_informative=10, random_state=42)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]

print("predict: ", model.predict(X_test))
print("predict proba: ", model.predict_proba(X_test))
