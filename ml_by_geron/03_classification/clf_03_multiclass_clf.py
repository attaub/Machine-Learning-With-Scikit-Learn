import sklearn
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from clf_utils import plot_digit, plot_digits
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from clf_utils import  plot_roc_curve

from sklearn.metrics import precision_recall_curve
from clf_utils import  plot_precision_recall_vs_threshold
from clf_utils import  plot_precision_vs_recall

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from scipy.ndimage.interpolation import shift
from sklearn.neighbors import KNeighborsClassifier

#################################################################
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
#################################################################
## Multilabel Classification

y_train_large = y_train >= 7
y_train_odd = y_train % 2 == 1
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

#################################################################
# Multioutput Classification

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index = 0
plt.subplot(121)
plot_digit(X_test_mod[some_index])
plt.subplot(122)
plot_digit(y_test_mod[some_index])
plt.show()

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

#################################################################
## Extra material
# Dummy (ie. random) classifier

"""
dmy_clf = DummyClassifier(strategy="prior")
y_probas_dmy = cross_val_predict(
    dmy_clf, X_train, y_train_5, cv=3, method="predict_proba"
)
y_scores_dmy = y_probas_dmy[:, 1]


fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
plot_roc_curve(fprr, tprr)
"""

#################################################################
# ## KNN classifier

knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)

y_knn_pred = knn_clf.predict(X_test)

accuracy_score(y_test, y_knn_pred)

def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)

plot_digit(shift_digit(some_digit, 5, 1, new=100))

X_train_expanded = [X_train]
y_train_expanded = [y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted_images = np.apply_along_axis(
        shift_digit, axis=1, arr=X_train, dx=dx, dy=dy
    )
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(y_train)

X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)
X_train_expanded.shape, y_train_expanded.shape

knn_clf.fit(X_train_expanded, y_train_expanded)

y_knn_expanded_pred = knn_clf.predict(X_test)

accuracy_score(y_test, y_knn_expanded_pred)

ambiguous_digit = X_test[2589]
knn_clf.predict_proba([ambiguous_digit])

plot_digit(ambiguous_digit)
#################################################################
