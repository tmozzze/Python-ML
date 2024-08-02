import numpy
import numpy as np
import pandas as pd

class MyLogReg():

    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric

        self.best_score = None


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

    def compute_roc_auc(self, y, y_hat):
        sorted_idx = np.argsort(y_hat)[::-1]
        y_true_sorted = y[sorted_idx]

        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        tpr = tps / np.sum(y_true_sorted)
        fpr = fps / np.sum(1 - y_true_sorted)

        # Добавим начальные и конечные точки (0, 0) и (1, 1)
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])

        auc = np.trapz(tpr, fpr)
        return round(auc, 10)


    def compute_metric(self, y, y_hat):
        if self.metric is not None:
            y_pred = (y_hat >= 0.5).astype(int)

            tp = np.sum((y_pred == 1) & (y == 1))
            fp = np.sum((y_pred == 1) & (y == 0))
            tn = np.sum((y_pred == 0) & (y == 0))
            fn = np.sum((y_pred == 0) & (y == 1))

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

            if self.metric == "accuracy":
                return accuracy
            if self.metric == "precision":
                return precision
            if self.metric == "recall":
                return recall
            if self.metric == "f1":
                return f1
            if self.metric == "roc_auc":
                return self.compute_roc_auc(y, y_hat)
            else:
                return None


    def fit(self, X, y, verbose=False):
        X = np.c_[np.ones(X.shape[0]), X]

        self.weights = np.ones(X.shape[1])



        for i in range(self.n_iter):
            y_hat = self.sigmoid(np.dot(X, self.weights))

            loss = self.compute_log_loss(y, y_hat)


            if verbose and (i % verbose == 0):
                if self.metric is not None:
                    metric_value = self.compute_metric(y, y_hat)
                    print(f"{i if i > 0 else 'start'} | loss: {loss:.2f} | learning rate: {self.learning_rate} | {self.metric}: {metric_value}")

                else:
                    print(f"{i if i > 0 else 'start'} | loss: {loss:.2f} | learning rate: {self.learning_rate}")

            error = y_hat - y
            gradient = np.dot(error, X) / len(y)

            self.weights -= self.learning_rate * gradient

        y_hat = self.sigmoid(np.dot(X, self.weights))

        if verbose:
            if self.metric is not None:
                metric_value = self.compute_metric(y, y_hat)
                print(f"final | loss: {loss:.2f} | learning rate: {self.learning_rate} | {self.metric}: {metric_value}")
            else:
                print(f"final | loss: {loss:.2f} | learning rate: {self.learning_rate}")

        self.best_score = self.compute_metric(y, y_hat)

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

    def get_best_score(self):
        return self.best_score


from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]



model = MyLogReg(n_iter=200, learning_rate=0.1, metric="roc_auc")

print(model)

model.fit(X, y, 10)

print("coef: ",model.get_coef())

print("best score: ", model.get_best_score())

X_test, y_test = make_classification(n_samples=400, n_features=14, n_informative=10, random_state=42)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)
X_test.columns = [f'col_{col}' for col in X_test.columns]


print("predict: ", np.sum(model.predict(X_test)))
print(f"predict proba: {np.mean(model.predict_proba(X_test)):.2f}")
