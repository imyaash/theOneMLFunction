# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:26:53 2023

@author: imyaash-admin

Module: Classification Candidate Model

This module contains a function to define and customize the hyperparameters of
various classification models available in the Scikit-learn library.
The function "classificationModelsParams" takes in a random state as input and
returns a list of tuples. Each tuple contains a classification model object,
a dictionary of its corresponding hyperparameters to be tuned,
and a string representing the evaluation metric to be used for cross-validation.

The available classification models are:
- LogisticRegression
- RidgeClassifier
- SGDClassifier
- GaussianNB
- MultinomialNB
- KNeighborsClassifier
- RadiusNeighborsClassifier
- GaussianProcessClassifier
- DecisionTreeClassifier
- RandomForestClassifier
- ExtraTreesClassifier
- GradientBoostingClassifier

Each classification model has a set of hyperparameters that can be customized using
a dictionary. The function returns a list of tuples, where each tuple contains
a classification model object, a dictionary of its corresponding hyperparameters to
be tuned, and a string representing the evaluation metric to be used for
cross-validation. The models and their respective hyperparameters are pre-defined
within the function.

This module can be used to easily obtain a set of regression models with their
corresponding hyperparameters to be tuned. This can be useful for selecting the
best classification model for a given dataset and optimizing its hyperparameters
for better performance.

"""

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier

def classificationCandidateModels(randomState):
    # Defining models for classification
    models = [
        (LogisticRegression(max_iter = 10000, n_jobs = -1, random_state = randomState), {
            "penalty"       : ["l1", "l2", "elasticnet", "none"],
            "C"             : [0.2, 0.4, 0.6, 0.8, 1.0]
            }, "accuracy"),

        (RidgeClassifier(random_state = randomState), {
            "alpha"         : np.linspace(0.01, 1.0, 10),
            "solver"        : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
            }, "accuracy"),

        (SGDClassifier(early_stopping = True, max_iter = 10000, n_jobs = -1, random_state = randomState), {
            "loss"          : ["hinge", "logloss", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty"       : ["l2", "l1", "elasticnet"],
            "alpha"         : np.linspace(0.000001, 0.1, 10),
            "learning_rate" : ["constant", "optimal", "invscaling", "adaptive"]
            }, "accuracy"),

        (GaussianNB(), {
            "var_smoothing" : np.linspace(0.00000000001, 0.0000001, 20)
            }, "accuracy"),

        (MultinomialNB(), {
            "alpha"         : np.linspace(0.0001, 1.0, 20)
            }, "accuracy"),

        (KNeighborsClassifier(n_jobs = -1), {
            "n_neighbors"   : np.arange(2, 22, 2),
            "weights"       : ["uniform", "distance"],
            "algorithm"     : ["auto", "ball_tree", "kd_tree", "brute"],
            }, "accuracy"),

        (RadiusNeighborsClassifier(n_jobs = -1), {
            "radius"        : np.linspace(0.0001, 1.0, 10),
            "weights"       : ["uniform", "distance"],
            "algorithm"     : ["auto", "ball_tree", "kd_tree", "brute"],
            }, "accuracy"),

        (GaussianProcessClassifier(n_jobs = -1, random_state = randomState), {
            "n_restarts_optimizer": np.arange(0, 10, 2)
            }, "accuracy"),

        (DecisionTreeClassifier(random_state = randomState), {
            "criterion"     : ["gini", "entropy", "log_loss"],
            "splitter"      : ["best", "random"],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8]
            }, "accuracy"),

        (RandomForestClassifier(n_jobs = -1, random_state = randomState), {
            "n_estimators"  : [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
            "criterion"     : ["gini", "entropy", "log_loss"],
            "oob_score"     : [True, False],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8],
            "max_samples"   : [0.3, 0.5, 0.7, 0.8, 0.9, None]
            }, "accuracy"),

        (ExtraTreesClassifier(n_jobs = -1, random_state = randomState), {
            "n_estimators"  :[50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
            "criterion"     : ["gini", "entropy", "log_loss"],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8],
            "max_samples"   : [0.3, 0.5, 0.7, 0.8, 0.9, None]
            }, "accuracy"),

        (GradientBoostingClassifier(), {
            "loss"          : ["log_loss", "exponential"],
            "learning_rate" : np.linspace(0.001, 0.25, 10),
            "n_estimators"  : [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
            "criterion"     : ["friedman_mse", "squared_error"],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8]
            }, "accuracy")
        ]
    return models

def classificationEnsembleModel(baseEstimators, randomState):
    model = ((AdaBoostClassifier(base_estimator = baseEstimators, random_state = randomState), {
    "n_estimators"          : [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
    "learning_rate"         : np.linspace(0.1, 2.5, 25),
    "loss"                  : ["linear", "square", "exponential"]
    }, "accuracy"))
    return model
