# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:25:06 2023

@author: imyaash-admin

Helper functions for theOneFunction (Machine Learning Model Selector, Builder & Hyper-Parameter Tuner).

"""

import random
from scipy.stats import ttest_ind, ks_2samp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, Normalizer, StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import RandomizedSearchCV

def findBestSubset(data, n_subsets, subsetSize, randomState):
    """
    Randomly selects a subset of data from a dataset n_subsets times,
    performs hypothesis testing to test how closely the subsets represent the dataset,
    and outputs the subset that best represents the dataset based on the p-value of the hypothesis test.

    Parameters:
        data (DataFrame): Dataset to select subsets from
        n_subsets (int): Number of subsets to randomly select
        subsetSize (int): Size of each subset
        randomState (int): Setting a seed value

    Returns:
        (DataFrame): Subset that best represents the dataset

    """
    bestSubset = None
    best_pvalue = 0

    # Setting a seed value
    random.seed(randomState)

    for i in range(n_subsets):
        # Randomly selecting a subset
        subsetIndices = random.sample(range(data.shape[0]), subsetSize)
        subset = data.iloc[subsetIndices]

        # Performing hypothesis testing
        t_stat, pvalue = ttest_ind(data, subset)

        # Checking if pvalue for the subset is better than the current best pvalue
        if np.mean(pvalue) > best_pvalue:
            bestSubset = subset
            best_pvalue = np.mean(pvalue)

    return bestSubset

def cleaner(data):
    """
    Cleans a data frame by removing columns with more than 30% NaNs/empty rows and
    imputing NaNs/empty rows with the median (if there are outliers) or the mean (otherwise).

    Parameters:
        data (DataFrame): The data frame to clean.

    Returns:
        DataFrame: The cleaned data frame.
    """
    # Calculate the percentage of NaNs/empty rows in each column
    nanPercentages = data.isna().mean() * 100

    # Identify columns with more than 30% NaNs/empty rows
    toDrop = nanPercentages[nanPercentages > 30].index.tolist()

    # Remove the identified columns
    data = data.drop(toDrop, axis=1)

    # Impute NaNs/empty rows with the median (if there are outliers) or the mean (otherwise)
    for col in data.columns:
        median = data[col].median()
        q1, q3 = np.percentile(data[col], [25, 75])
        iqr = q3 - q1
        lBound = q1 - (1.5 * iqr)
        uBound = q3 + (1.5 * iqr)
        if np.isnan(lBound) or np.isnan(uBound):
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(median, inplace=True)

    return data

def selector(data, targetVar, isContinuous, randomState):
    """
    Selects the 5 most important features from a given data using random forest
    based feature selection.

    Parameters:
        data : pandas.DataFrame
            Input data containing the features and target variable.
        targetVar : str
            Name of the target variable in the input data.
        isContinuous : bool
            Whether the target variable is continuous or categorical.
        randomState : int
            Setting a seed value.

    Returns:
        data : pandas.DataFrame
            A DataFrame containing the selected features and the target variable.
        selected_features : list of str
            A list of the selected feature names.

    """
    # Splitting the data into features (X) and target variable (y)
    X = data.drop(targetVar, axis = 1)
    y = data[targetVar]

    # Creating a Random Forest Model
    if isContinuous:
        model = RandomForestRegressor(random_state = randomState)
    else:
        model = RandomForestClassifier(random_state = randomState)

    # Fitting the model on the dataset
    model.fit(X, y)

    # Getting the feature importances
    featureImportances = pd.Series(model.feature_importances_, index = X.columns)

    # Sortting the feature importance in descending order
    sortedImportances = featureImportances.sort_values(ascending = False)

    # Selecting the top 5 features
    selectedFeatures = sortedImportances.head(5).index.tolist()

    # Adding target variable to the selected features
    selectedFeatures.append(targetVar)

    return data[selectedFeatures]

def encoder(data):
    """
    Encodes non-numeric columns of a given pandas DataFrame using ordinal encoding.

    Parameters:
    -----------
    data : pandas DataFrame
        The input DataFrame to be encoded.

    Returns:
    --------
    pandas DataFrame
        The encoded DataFrame, with non-numeric columns transformed using ordinal encoding.

    Notes:
    ------
    1. If no non-numeric columns are found in the input DataFrame, the function returns the original DataFrame.
    2. The function modifies the input DataFrame in place and does not create a new copy.

    """
    # Checking if any column is not numerical
    nonNumCols = data.select_dtypes(exclude = ["int", "float"]).columns

    if len(nonNumCols) == 0:
        return data

    # Initialising the ordinal encoder
    ordinalEncoder = OrdinalEncoder()

    # Fitting and transforming the encoder on the non numerical columns
    data[nonNumCols] = ordinalEncoder.fit_transform(data[nonNumCols])

    return data

def normaliser(X):
    """
    This function performs standard scaling on the input data and returns the Normalised data.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data containing the features.

    Returns
    -------
    ndarray or sparse matrix, shape (n_samples, n_features)
        Normalised input data.

    """
    norm = Normalizer()
    return norm.fit_transform(X)

def classBalanceChecker(data, targetVar):   # Unused
    """
    Checks if the classes in a dataset are balanced by calculating the class ratio and returns a class_weight dictionary
    if the ratio is greater than 5.

    Parameters:
    -----------
    data : pandas DataFrame
        The input dataset.
    targetVar : str
        The name of the column that contains the target variable.

    Returns:
    --------
    dict or None
        Returns a dictionary of class weights if the class ratio is greater than 5, otherwise returns None.

    Raises:
    -------
    ValueError
        If the number of unique classes in the dataset is less than 2.
    """
    # Getting the count of unique classes in the data
    numClasses = len(data[targetVar].unique())

    # Raising exception if data has less than 2 class
    if numClasses < 2:
        raise ValueError("Data should have 2 or more classes.")

    # Calculating the class ratio
    classRatios = {}
    for i in range(numClasses):
        classI = data[data[targetVar]] == data[targetVar].unique()[i]
        classRatios[i] = sum(classI) / len(data)

    # Checking if the class ratio is greater than 5
    if max(classRatios.values()) / min(classRatios.values()) > 5:
        # Calculating class weights
        total = sum(classRatios.values())
        classWeights = {}
        for i in range(numClasses):
            classWeights[i] = total / (numClasses * classRatios[i])
        return classWeights
    else:
        return None

def selector_v2(data, targetVar, isContinuous, randomState, k = 5):
    """
    Performs mutual information-based feature selection on the input data.

    Parameters:
        data : pandas.DataFrame
            Input data containing the features and target variable.
        targetVar : str
            Name of the target variable in the input data.
        isContinuous : bool
            Whether the target variable is continuous or categorical.
        randomState : int
            Setting a seed value.
        k   : int, default = 5
            Number of top features to select.

    Returns:
        data : pandas.DataFrame
            A DataFrame containing the selected features and the target variable.
        selected_features : list of str
            A list of the selected feature names.

    """

    # Separating the features and target variable
    features = data.drop(targetVar, axis = 1)
    target = data[targetVar]

    # Performing mutual information-based feature selection
    if isContinuous:
        mi = mutual_info_regression(features, target, random_state=randomState)
    else:
        mi = mutual_info_classif(features, target, random_state=randomState)

    # Ranking the features by their mutual information with the target variable
    ranked_features = pd.Series(mi, index=features.columns)
    ranked_features.sort_values(ascending=False, inplace=True)

    # Selecting the top k features
    selected_features = ranked_features.index[:k]

    # Creating a new DataFrame with the selected features and the target variable
    selected_data = pd.concat([features[selected_features], target], axis=1)

    return selected_data, selected_features

def trainer(models, sub, targetVar, randomState):
    """
    Trains each model using cross-validation to find the best hyper-parameters.

    Parameters:
        models : list of tuples
            A list of tuples containing the model, hyperparameters to search over, and scoring method.
        sub : pandas.DataFrame
            Input data containing the subset of features and target variable.
        targetVar : str
            Name of the target variable column in the subset data.
        randomState : int
            Setting a seed value.

    Returns:
        bestModel : sklearn.model_selection._search.RandomizedSearchCV
            The best model based on the cross-validation results.
        bestScore : float
            The mean cross-validation score of the best model.

    """
    # Splitting the stratified subset into features and target
    features = sub.drop(targetVar, axis = 1)
    target = sub[targetVar]
    
    # Training each model and finding the best hyper-parameters by cross-validation
    bestModel = None
    bestScore = 0
    for model, hyperparams, scoring in models:
        tuner = RandomizedSearchCV(model, hyperparams, cv = 5, n_jobs = -1, n_iter = 25, scoring = scoring, verbose = 2, random_state = randomState)
        # Checking if the model is not tree-based and using scaled data for not-tree based models
        if type(model).__name__ in ["DecisionTreeRegressor", "DecisionTreeClassifier", "RandomForestRegressor", "RandomForestClassifier", "ExtraTreesRegressor", "ExtraTreesClassifier", "GradientBoostingRegressor", "GradientBoostingClassifier"]:
            tuner.fit(features, target)
        elif type(model).__name__ in ["LinearRegression", "Ridge", "Lasso", "SGDRegressor", "ElasticNet", "BayesianRidge", "KNeighborsRegressor", "RadiusNeighborsRegressor", "GaussianProcessRegressor"]:
            normalisedFeatures = normaliser_v2(features, "standard")
            tuner.fit(normalisedFeatures, target)
        else:
            normalisedFeatures = normaliser_v2(features, "minmax")
            tuner.fit(normalisedFeatures, target)
        score = tuner.best_score_
        if np.nanmean(score) > bestScore:
            bestModel = tuner
            bestScore = np.nanmean(score)
    return bestModel, bestScore

def normaliser_v2(X, method):
    """
    This function performs either standard scaling or min-max scaling on the input data
    and returns the normalised data.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data containing the features.
    method  : str {"standard", "minmax"}
        Normalization method to be applied.

    Returns
    -------
    ndarray or sparse matrix, shape (n_samples, n_features)
        Normalised input data.

    Raises
    ------
    ValueError
        If method is not one of "standard" or "minmax".

    """
    if method == "standard":
        normaliser = StandardScaler()
    elif method == "minmax":
        normaliser = MinMaxScaler()
    else:
        raise ValueError("method must be one of 'standard' or 'minmax'.")

    return normaliser.fit_transform(X)

def cleaner_v2(data):
    """
    Cleans a data frame by removing columns with more than 30% NaNs/empty rows,
    imputing NaNs/empty rows with the median (if there are outliers) or the mean (otherwise),
    and dropping any duplicate rows.

    Parameters:
        data (DataFrame): The data frame to clean.

    Returns:
        DataFrame: The cleaned data frame.
    """
    # Identify duplicate rows and drop them
    data.drop_duplicates(inplace=True)

    # Calculate the percentage of NaNs/empty rows in each column
    nanPercentages = data.isna().mean() * 100

    # Identify columns with more than 30% NaNs/empty rows
    toDrop = nanPercentages[nanPercentages > 30].index.tolist()

    # Remove the identified columns
    data = data.drop(toDrop, axis=1)

    # Impute NaNs/empty rows with the median (if there are outliers) or the mean (otherwise)
    for col in data.columns:
        median = data[col].median()
        q1, q3 = np.percentile(data[col], [25, 75])
        iqr = q3 - q1
        lBound = q1 - (1.5 * iqr)
        uBound = q3 + (1.5 * iqr)
        if np.isnan(lBound) or np.isnan(uBound):
            data[col].fillna(data[col].mean(), inplace=True)
        else:
            data[col].fillna(median, inplace=True)

    return data

def subsetFinder(data, nSubsets, subsetSize, randomState):
    bestSubset = None
    bestAvgPValue = 0

    # Setting a seed value
    random.seed(randomState)

    for i in range(nSubsets):
        # Randomly selecting a subset
        subsetIndices = random.sample(range(data.shape[0]), subsetSize)
        subset = data.iloc[subsetIndices]

        # Compute p-value for each column
        pValues = []
        for col in data.columns:
            _, pValue = ks_2samp(data[col], subset[col])
            pValues.append(pValue)

        # Checking if the average p-value of the subset is better than the current best average p-value
        if np.mean(pValues) > bestAvgPValue:
            bestSubset = subset
            bestAvgPValue = np.mean(pValues)

    return bestSubset

def retrainer(bestModel, data, targetVar):
    """
    Retrains the best model using the non-scaled data for tree-based models and
    the standard or minmax scaled data for other models.

    Parameters:
    bestModel : sklearn.model_selection._search.RandomizedSearchCV
        The best model based on the cross-validation results.
    data : pandas.DataFrame
        Input data containing the features and target variable.
    targetVar : str
        The name of the target variable in the data.

    Returns:
    bestModel : sklearn.model_selection._search.RandomizedSearchCV
        The retrained best model based on the cross-validation results.
    bestScore : float
        The mean cross-validation score of the retrained model.
    bestParams : dict
        The best hyper-parameters found by cross-validation of the retrained model.

    """

    # Splitting the data into features and target
    features = data.drop(targetVar, axis = 1)
    target = data[targetVar]

    # Re-fitting the best model with the whole dataset, while using the non scaled data for the tree-based models
    if type(bestModel.best_estimator_).__name__ in ["DecisionTreeRegressor", "DecisionTreeClassifier", "RandomForestRegressor", "RandomForestClassifier", "ExtraTreesRegressor", "ExtraTreesClassifier", "GradientBoostingRegressor", "GradientBoostingClassifier"]:
        bestModel.fit(features, target)
    elif type(bestModel.best_estimator_).__name__ in ["LinearRegression", "Ridge", "Lasso", "SGDRegressor", "ElasticNet", "BayesianRidge", "KNeighborsRegressor", "RadiusNeighborsRegressor", "GaussianProcessRegressor"]:
        normalisedFeatures = normaliser_v2(features, "standard")
        bestModel.fit(normalisedFeatures, target)
    else:
        normalisedFeatures = normaliser_v2(features, "minmax")
        bestModel.fit(normalisedFeatures, target)

    bestScore = bestModel.best_score_
    bestParams = bestModel.best_params_

    return bestModel, bestScore, bestParams
