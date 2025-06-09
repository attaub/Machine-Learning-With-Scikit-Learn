import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.base import BaseEstimator

from clf_utils import plot_digit, plot_digits

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#################################################################
# # Training a Binary Classifier

y_train_5 = y_train == 5
y_test_5 = y_test == 5

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#################################################################
# Measuring Accuracy Using Cross-Validation

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
print()
print(
    cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
)
