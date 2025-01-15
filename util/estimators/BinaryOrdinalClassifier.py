import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator


class BinaryOrdinalClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, classifier,breakpoints, params = {}):
        self.breakpoints = []
        self.model_list = []
        self.classifier = classifier
        self.breakpoints = breakpoints
        self.params = params

    def fit(self, X, y):
        self.model_list = []
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        for b in self.breakpoints:
            # Create new binary label
            y_b = y.apply(lambda l: 1 if l > b else 0)
            model = self.classifier(**self.params)
            # Fit binary classifier
            model.fit(X, y_b)
            # Append binary classifier to list of binary classifiers
            self.model_list.append(model)

    def predict(self, X):
        y_proba_result = self.predict_proba(X)
        y_result = np.argmax(y_proba_result, axis=1)
        return y_result

    def predict_proba(self, X):
        y_probas = np.zeros(shape=(X.shape[0], len(self.breakpoints) + 1))
        for i in range(0, len(self.breakpoints)+1):
            if i == 0:
                y = 1 - self.model_list[i].predict_proba(X)[:, 1]
                y_probas[:, i] = y.flatten()
            elif i == len(self.breakpoints):
                y = self.model_list[i-1].predict_proba(X)[:, 1]
                y_probas[:, i] = y.flatten()
            else:
                # Version from https://www.mathematik.uni-marburg.de/~eyke/publications/ov.pdf
                y = np.maximum(
                    self.model_list[i - 1].predict_proba(X)[:, 1] - self.model_list[i].predict_proba(X)[:, 1], 0)
                y_probas[:, i] = y.flatten()
        return y_probas
