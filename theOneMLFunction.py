# -*- coding: utf-8 -*-
"""
 ____  _      _____      ____  _____  ____  ____    _      _    
/  _ \/ \  /|/  __/     / ___\/__ __\/  _ \/  __\  / \__/|/ \   
| / \|| |\ |||  \ _____ |    \  / \  | / \||  \/|  | |\/||| |   
| \_/|| | \|||  /_\____\\___ |  | |  | \_/||  __/  | |  ||| |_/\
\____/\_/  \|\____\     \____/  \_/  \____/\_/     \_/  \|\____/
                                                                

 _______________________________________________________________
|                                                               |
|                             tOMLf                             |
|           the One-stop Machine Learning Function              |
|______________________________________________________________ |
|                                                               |
|  - Data cleaning                                              |
|  - Feature selection                                          |
|  - Model training                                             |
|  - Hyperparameter tuning                                      |
|  - Result printing                                            |
|  - Model ensembling                                           |
|______________________________________________________________ |

"""
"""
Created on Wed Mar  1 00:22:45 2023

@author: imyaash-admin

This Python function provides an end-to-end machine learning solution for both
regression and classification problems. It automates data preprocessing, feature
selection, model training and tuning, and generates the best model with the
highest cross-validation score.

"""

"""
# Need to look into implementing classBalancerChecker
# Need to look into providing a more detailed output:
    # Implementing the metrics functions.
    # Feature importance / coefficeints plots
# Need to look into implementing Multi-Label Classification/Regression

"""

from utility import encoder, cleaner_v2, selector_v2, subsetFinder, trainer_v2, retrainer_v2, ensembler, compareNselect
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
    else:
        selectedFeatures = None

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

    # Using the trainer_v2 to train the models and tune hyper-parameters
    bestModel, bestScore, top3Models = trainer_v2(models, sub, targetVar, randomState)
    
    # Using the ensembler to combine the top 3 models and select the best ensemble model
    bestEnsembleModel, bestEnsembleScore = ensembler(top3Models, sub, targetVar, isContinuous, randomState)
    
    # Selecting the best model with compareNselect
    bestModel, bestScore = compareNselect(bestModel, bestScore, bestEnsembleModel, bestEnsembleScore)

    # Retraining the best model on the whole data with retrainer_v2
    bestModel, bestScore, bestParams = retrainer_v2(bestModel, data, targetVar)

    # Printing the best model's name, it's hyper-parameters and it's score
    print(type(bestModel.best_estimator_).__name__)
    print(bestParams)
    print(bestScore)

    return bestModel.best_estimator_, selectedFeatures
