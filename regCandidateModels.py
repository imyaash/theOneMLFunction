# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:09:19 2023

@author: imyaash-admin

Module: Regression Candidate Model

This module contains a function to define and customize the hyperparameters of 
various regression models available in the Scikit-learn library.
The function "regressionModelsParams" takes in a random state as input and
returns a list of tuples. Each tuple contains a regression model object,
a dictionary of its corresponding hyperparameters to be tuned,
and a string representing the evaluation metric to be used for cross-validation.

The available regression models are:
- LinearRegression
- Ridge
- Lasso
- SGDRegressor
- ElasticNet
- BayesianRidge
- KNeighborsRegressor
- RadiusNeighborsRegressor
- GaussianProcessRegressor
- DecisionTreeRegressor
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor

Each regression model has a set of hyperparameters that can be customized using
a dictionary. The function returns a list of tuples, where each tuple contains
a regression model object, a dictionary of its corresponding hyperparameters to
be tuned, and a string representing the evaluation metric to be used for
cross-validation. The models and their respective hyperparameters are pre-defined
within the function.

This module can be used to easily obtain a set of regression models with their
corresponding hyperparameters to be tuned. This can be useful for selecting the
best regression model for a given dataset and optimizing its hyperparameters
for better performance.

"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor,ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

def regressionCandidateModels(randomState):
    # Defining models for regression
    models = [
        (LinearRegression(), {
            "n_jobs"        : [-1]
            }, "r2"),
        
        (Ridge(random_state = randomState), {
            "alpha"         : np.linspace(0.01, 1.0, 10),
            "solver"        : ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
            }, "r2"),
        
        (Lasso(max_iter = 10000, random_state = randomState), {
            "alpha"         : np.linspace(0.01, 1.0, 10),
            "selection"     : ["cyclic", "random"]
            }, "r2"),
        
        (SGDRegressor(early_stopping = True, max_iter = 10000, random_state = randomState), {
            "loss"          : ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty"       : ["l2", "l1", "elasticnet"],
            "alpha"         : np.linspace(0.000001, 0.1, 10),
            "learning_rate" : ["constant", "optimal", "invscaling", "adaptive"]
            }, "r2"),
        
        (ElasticNet(max_iter = 10000, random_state = randomState), {
            "alpha"         : np.linspace(0.01, 1.0, 10),
            "l1_ratio"      : np.linspace(0.01, 1.0, 10),
            "selection"     : ["cyclic", "random"]
            }, "r2"),
        
        (BayesianRidge(n_iter = 10000), {
            "alpha_1"       : np.linspace(0.0000001, 0.5, 10),
            "alpha_2"       : np.linspace(0.0000001, 0.5, 10),
            "lambda_1"      : np.linspace(0.0000001, 0.5, 10),
            "lambda_2"      : np.linspace(0.0000001, 0.5, 10)
            }, "r2"),
        
        (KNeighborsRegressor(n_jobs = -1), {
            "n_neighbors"   : np.arange(2, 22, 2),
            "weights"       : ["uniform", "distance"],
            "algorithm"     : ["auto", "ball_tree", "kd_tree", "brute"],
            }, "r2"),
        
        (RadiusNeighborsRegressor(n_jobs = -1), {
            "radius"        : np.linspace(0.0001, 5.0, 20),
            "weights"       : ["uniform", "distance"],
            "algorithm"     : ["auto", "ball_tree", "kd_tree", "brute"],
            }, "r2"),
        
        (GaussianProcessRegressor(random_state = randomState), {
            "alpha"         : np.linspace(0.00000000001, 0.0001, 10),
            "n_restarts_optimizer": np.arange(0, 10, 2)             
            }, "r2"),
        
        (DecisionTreeRegressor(random_state = randomState), {
            "criterion"     : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter"      : ["best", "random"],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8]
            }, "r2"),
        
        (RandomForestRegressor(n_jobs = -1, random_state = randomState), {
            "n_estimators"  : [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
            "criterion"     : ["squared_error", "absolute_error", "poisson"],
            "oob_score"     : [True, False],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8],
            "max_samples"   : [0.3, 0.5, 0.7, 0.8, 0.9, None]
            }, "r2"),
        
        (ExtraTreesRegressor(n_jobs = -1, random_state = randomState), {
            "n_estimators"  : [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
            "criterion"     : ["squared_error", "absolute_error"],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8]
            }, "r2"),
        
        (GradientBoostingRegressor(n_iter_no_change = 10, random_state = randomState), {
            "loss"          : ["squared_error", "absolute_error", "huber", "quantile"],
            "learning_rate" : np.linspace(0.001, 0.25, 10),
            "n_estimators"  : [50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500],
            "criterion"     : ["friedman_mse", "squared_error"],
            "ccp_alpha"     : [0.0, 0.2, 0.4, 0.6, 0.8]
            }, "r2")
        ]
    return models