import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
#################################################################
## MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

y_train_5 = y_train == 5
y_test_5 = y_test == 5

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#################################################################
## Performance Measures
### Confusion Matrix

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

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
y_train_perfect_predictions = y_train_5  # pretend we reached perfection

confusion_matrix(y_train_5, y_train_pred)
confusion_matrix(y_train_5, y_train_perfect_predictions)

#################################################################
### Precision and Recall

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
cm_sk = confusion_matrix(y_train_5, y_train_pred)
cm=cm_sk.copy()
cm_py=cm[1, 1] / (cm[0, 1] + cm[1, 1])
precison_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
f1_sk = f1_score(y_train_5, y_train_pred)
f1_py = cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)

print()
print("Confusion Matrix\n")
print("\t",cm_py)
print("\t",cm_sk)
print("precison")
print("\t",precison_1)
print("f1-Score")
print("\t",f1_py)
print("\t",f1_sk)

#################################################################
# ## Precision/Recall Trade-off

y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = y_scores > threshold

y_some_digit_pred

threshold = 8000
y_some_digit_pred = y_scores > threshold
y_some_digit_pred

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

plt.figure(figsize=(8, 4))  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot(
    [threshold_90_precision, threshold_90_precision], [0.0, 0.9], "r:"
)  # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")  # Not shown
plt.plot(
    [-50000, threshold_90_precision],
    [recall_90_precision, recall_90_precision],
    "r:",
)  # Not shown
plt.plot([threshold_90_precision], [0.9], "ro")  # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")  # Not shown
plt.show()

(y_train_pred == (y_scores > 0)).all()

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0.0, 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
plt.show()

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
threshold_90_precision

y_train_pred_90 = y_scores >= threshold_90_precision
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

#################################################################
## The ROC Curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.figure(figsize=(8, 6))  # Not shown
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]  # Not shown
plt.plot([fpr_90, fpr_90], [0.0, recall_90_precision], "r:")  # Not shown
plt.plot(
    [0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:"
)  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")  # Not shown
plt.show()

roc_auc_score(y_train_5, y_scores)

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(
    y_train_5, y_scores_forest
)

recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([fpr_90, fpr_90], [0.0, recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
plt.plot([fpr_90, fpr_90], [0.0, recall_for_forest], "r:")
plt.plot([fpr_90], [recall_for_forest], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
plt.show()

roc_auc_score(y_train_5, y_scores_forest)

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)

recall_score(y_train_5, y_train_pred_forest)
