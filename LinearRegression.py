import random

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

class MyLineReg():

    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.best_score = None

    def __str__(self):
        return (f"MyLineReg class: n_iter={self.n_iter},  learning_rate={self.learning_rate}"
                f"metric={self.metric}, "
                f"reg={self.reg}, l1_coef={self.l1_coef}, l2_coef={self.l2_coef}, sgd_sample={self.sgd_sample}, random_state={self.random_state}")
    def _compute_metric(self, y, y_pred):
        error = y - y_pred
        if self.metric == "mse":
            return np.mean(error ** 2)
        elif self.metric == "mae":
            return np.mean(np.abs(error))
        elif self.metric == "rmse":
            return np.sqrt(np.mean(error ** 2))
        elif self.metric == "mape":
            return np.mean(np.abs(error / y)) * 100
        elif self.metric == "r2":
            ss_res = np.sum(error ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            return None

    def fit(self, X, y, verbose=False):
        X = np.c_[np.ones(X.shape[0]), X]
        #init weights
        self.weights = np.ones(X.shape[1])
        #random seed
        random.seed(self.random_state)


        for i in range(1, self.n_iter+1):
            #Dynamic Learning_rate
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate

            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float) and 0.0 < self.sgd_sample <= 1.0:
                    sgd_sample_size = int(self.sgd_sample * X.shape[0])
                elif isinstance(self.sgd_sample, int) and self.sgd_sample > 0:
                    sgd_sample_size = self.sgd_sample
                else:
                    raise ValueError("sgd_sample should be a positive or a float between 0.0 to 1.0")

                sample_rows_idx = random.sample(range(X.shape[0]), sgd_sample_size)
                X_sample = X[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]

            else:
                X_sample = X
                y_sample = y

            y_pred = X.dot(self.weights)
            y_sample_pred = X_sample.dot(self.weights)

            error = y - y_pred
            mse = np.mean(error ** 2)

            metric_value = self._compute_metric(y, y_pred)
            if self.metric is not None:
                self.best_score = metric_value

            #Regularization
            if self.reg is not None:
                if self.reg == "l1":
                    reg_loss = self.l1_coef * np.sum(np.abs(self.weights))
                elif self.reg == "l2":
                    reg_loss = self.l2_coef * np.sum(self.weights ** 2)
                elif self.reg == "elasticnet":
                    reg_loss = (self.l1_coef * np.sum(np.abs(self.weights))) + (self.l2_coef * np.sum(self.weights ** 2))
                else:
                    reg_loss = 0.0

                total_loss = mse + reg_loss
            else:
                total_loss = mse

            if verbose and (i % verbose == 0 or i == 1):
                if self.metric is not None:
                    print(f"{i if i > 0 else 'start'} | loss: {total_loss:.2f} | {self.metric}: {metric_value:.2f} | learning rate: {lr:.6f}")
                else:
                    print(f"{i if i > 0 else 'start'} | loss: {total_loss:.2f} | learning rate: {lr:.6f}")

            error_sample = y_sample - y_sample_pred
            gradient = -2 * X_sample.T.dot(error_sample) / len(y_sample)

            if self.reg is not None:

                if self.reg == "l1":
                    gradient += self.l1_coef * np.sign(self.weights)
                elif self.reg == "l2":
                    gradient += 2 * self.l2_coef * self.weights
                elif self.reg == "elasticnet":
                    gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            self.weights -= lr * gradient



        if verbose:
            if self.metric is not None:
                print(f"final | loss: {total_loss:.2f} | {self.metric}: {metric_value:.2f} | learning rate: {lr:.6f}")
            else:
                print(f"final | loss: {total_loss:.2f} | learning rate: {lr:.6f}")

        yy_pred = X.dot(self.weights)
        metric_value = self._compute_metric(y, yy_pred)

        if self.metric is not None:
            self.best_score = metric_value

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.weights)

    def get_best_score(self):
        return self.best_score




X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]


# Создаем и обучаем модель
dynamic_lr = lambda iter: 0.5 * (0.85 ** iter)
model = MyLineReg(n_iter=100, learning_rate=.1, metric="mape", reg="elasticnet", l1_coef=.1, l2_coef=.1, sgd_sample=.1) #, reg="l1", l1_coef=.1, l2_coef=.1
model.fit(X, y, 5)

print(f"Best score: {model.get_best_score():.2f}%")

coefs = model.get_coef()
print("Coefficients:", coefs)



X_test, y_test = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5, random_state=42)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]

# Получение предсказаний
predictions = model.predict(X_test)

# Вывод суммы предсказаний
print("Sum of predictions:", predictions.sum())
