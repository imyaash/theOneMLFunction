# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 00:22:45 2023

@author: imyaash-admin
"""

from utility import encoder, cleaner_v2, selector_v2, subsetFinder, trainer, retrainer
from regCandidateModels import regressionCandidateModels
from clfCandidateModels import classificationCandidateModels

def theOneMLFunction(data, targetVar, isContinuous, randomState = None):
    """
    The One Machine Learning (ML) Function performs a full ML pipeline,
    including data cleaning, feature selection, model training, tuning hyper-parameters,
    and retraining the best model on the entire dataset.

    Parameters:
        data : pandas.DataFrame
            Input data containing the features and target variable.
        targetVar : str
            Name of the target variable column in the input data.
        isContinuous : bool
            Boolean indicating whether the target variable is continuous or categorical.
        randomState : int, optional
            Setting a seed value, by default None.
    
    Returns:
        bestModel.best_estimator_ : sklearn model
            The best performing model.
        selectedFeatures : list
            The list of selected features.
    
    Raises:
        ValueError: If the target variable is not present in the input data.

    """
    
    # Checking if the target variable is present in the data
    if targetVar not in data.columns:
        raise ValueError(f"Target variable {targetVar} not found in the input data.")
    
    # Encoding the data
    data = encoder(data)
    
    # Cleaning the data
    data = cleaner_v2(data)
    
    # Selecting the 5 most important features
    if data.shape[1] > 5:
        data, selectedFeatures = selector_v2(data, targetVar, isContinuous, randomState)
    
    # Selecting a stratified subset of the input data if it has more than 2000 observations
    if data.shape[0] > 2000:
        sub = subsetFinder(data, 50, 2000, randomState)
    else:
        sub = data
    
    # Calling the list containing defined model, their hyper-parameter grid as well as scoring criteria
    if isContinuous:
        models = regressionCandidateModels(randomState)
    else:
        models = classificationCandidateModels(randomState)
    
    # Using the trainer to train the models and tune hyper-parameters
    bestModel, bestScore = trainer(models, sub, targetVar, randomState)
    
    # Retraining the best model on the whole data
    bestModel, bestScore, bestParams = retrainer(bestModel, data, targetVar)
    
    # Printing the best model's name, it's hyper-parameters and it's score
    print(type(bestModel.best_estimator_).__name__)
    print(bestParams)
    print(bestScore)
    
    return bestModel.best_estimator_, selectedFeatures