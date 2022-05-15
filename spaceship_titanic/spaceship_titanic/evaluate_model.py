# This script evaluates the trained model pipeline
# in various ways with train data

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score, precision_recall_curve, roc_curve, roc_auc_score

from config import PATH_DATA_FOLDER, PATH_MODEL_FOLDER


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)", fontsize=16)
    plt.grid(True)


if __name__ == "__main__":
    pipeline = joblib.load(PATH_MODEL_FOLDER + "pipeline.pkl")
    # decision_function vs. predict_proba
    if hasattr(pipeline, "decision_function"):
        method = "decision_function"
    else:
        method = "predict_proba"

    # train data
    train_data = pd.read_csv(PATH_DATA_FOLDER + "train.csv")
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    # cross validation confusion matrix
    pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=3)
    print("Confusion matrix with 3-fold CV: ")
    print(confusion_matrix(y_train, pred_cv))

    # Precision, recall and f1 score
    print(f"Precision score: {precision_score(y_train, pred_cv)}")
    print(f"Recall score: {recall_score(y_train, pred_cv)}")
    print(f"F1 score: {f1_score(y_train, pred_cv)}")

    # Decision function for threshold with precision recall curve
    pred_scores = cross_val_predict(
        pipeline, X_train, y_train, cv=3, method=method
    )
    if method == "predict_proba":
        pred_scores = pred_scores[:, 1]
    prec, rec, thresh = precision_recall_curve(y_train, pred_scores)
    plot_precision_recall_vs_threshold(prec, rec, thresh)
    plt.show()

    # New threshold
    threshold_new = 0.4  # -0.71
    pred_new = (pred_scores >= threshold_new)
    print(f"Precision score (new thrsh): {precision_score(y_train, pred_new)}")
    print(f"Recall score (new thrsh): {recall_score(y_train, pred_new)}")
    print(f"F1 score (new thrsh): {f1_score(y_train, pred_new)}")

    # ROC curve
    fpr, tpr, thresh = roc_curve(y_train, pred_scores)
    plot_roc_curve(fpr, tpr)
    plt.title(f"AUC: {roc_auc_score(y_train, pred_scores)}")
    plt.show()

    # test data
    test_data = pd.read_csv(PATH_DATA_FOLDER + "test.csv")
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    print(f"Testing score: {pipeline.score(X_test, y_test)}")
