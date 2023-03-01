# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:50:46 2023

@author: imyaash-admin

The module classification_and_regression_metrics contains two functions: classificationResult and regressionResult.

These functions are used to compute and print various classification and regression metrics for a given model.

"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def classificationResult(x_test, y_true, clf):
    """
    The classificationResult function takes three arguments:

    x_test: array-like, shape (n_samples, n_features) - The test input samples.
    y_true: array-like, shape (n_samples,) - The true labels for x_test.
    clf: object - The classification model.
    
    This function first makes predictions with the classification model,
    then plots the confusion matrix and ROC curve. After that, it computes and
    prints various classification metrics, including accuracy, precision, recall,
    F1-score, and ROC AUC score.
    
    """
    # Making predictions with classification model
    y_pred = clf.predict(x_test)
    
    # Storing confusion matrix in a variable
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Plotting confusion matrix and storing it in a variable
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
    fig_conf_matrix = disp.plot()

    # Computing ROC AUC
    if len(np.unique(y_true)) == 2:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, clf.predict_proba(x_test)[:, 1])
        roc_auc = metrics.roc_auc_score(y_true, clf.predict_proba(x_test)[:, 1])

        # Plotting ROC curve and storing it in a variable
        fig_roc_curve, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
    else:
        fig_roc_curve = None

    # Computing classification metrics and storing them in a variable
    metrics_dict = {}
    metrics_dict["Model Name"] = type(clf).__name__
    metrics_dict["Accuracy Score"] = metrics.accuracy_score(y_true, y_pred)
    metrics_dict["Balanced Accuracy Score"] = metrics.balanced_accuracy_score(y_true, y_pred)
    metrics_dict["Precision Score"] = metrics.precision_score(y_true, y_pred, average='macro')
    metrics_dict["Recall Score"] = metrics.recall_score(y_true, y_pred, average='macro')
    metrics_dict["F1 Score"] = metrics.f1_score(y_true, y_pred, average='macro')
    if len(np.unique(y_true)) == 2:
        metrics_dict["ROC AUC Score"] = metrics.roc_auc_score(y_true, clf.predict_proba(x_test)[:, 1])
    else:
        metrics_dict["ROC AUC Score"] = None
    metrics_dict["Confusion Matrix"] = conf_matrix

    return metrics_dict, fig_conf_matrix, fig_roc_curve



def regressionResult(x_test, y_true, rm):
    """
    The regressionResult function takes three arguments:

    x_test: array-like, shape (n_samples, n_features) - The test input samples.
    y_true: array-like, shape (n_samples,) - The true labels for x_test.
    rm: object - The regression model.
    
    This function first makes predictions with the regression model.
    Then, it computes and returns various regression metrics as a dictionary,
    including explained variance, max error, mean absolute error,
    mean squared error, root mean squared error, median absolute error,
    R-squared, and various other deviation scores.
    
    """
    # Making predictions with regression model
    y_pred = rm.predict(x_test)
    
    # Printing classification Metrics
    metric_dict = {
        "Model": type(rm).__name__,
        "Explained Variance": metrics.explained_variance_score(y_true, y_pred),
        "Max Error": metrics.max_error(y_true, y_pred),
        "Mean Absolute Error": metrics.mean_absolute_error(y_true, y_pred),
        "Mean Squared Error": metrics.mean_squared_error(y_true, y_pred, squared=True),
        "Root Mean Squared Error": metrics.mean_squared_error(y_true, y_pred, squared=False),
        "Mean Squared log Error": metrics.mean_squared_log_error(y_true, y_pred),
        "Median Absolute Error": metrics.median_absolute_error(y_true, y_pred),
        "R Squared": metrics.r2_score(y_true, y_pred),
        "Mean Poisson Deviance": metrics.mean_poisson_deviance(y_true, y_pred),
        "Mean Gamma Deviance": metrics.mean_gamma_deviance(y_true, y_pred),
        "Mean Absolute Percentage Error": metrics.mean_absolute_percentage_error(y_true, y_pred),
        "D Squared Absolute Error Score": metrics.d2_absolute_error_score(y_true, y_pred),
        "D Squared Pinball Score": metrics.d2_pinball_score(y_true, y_pred),
        "D Squared Tweedie Score": metrics.d2_tweedie_score(y_true, y_pred)
    }
    
    return metric_dict