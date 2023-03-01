# theOneMLFunction
This Python script provides a one-stop machine learning function that can handle both regression and classification tasks. It includes data cleaning, feature selection, model training, hyperparameter tuning, and result printing.

This Python module includes theOneMLFunction, which is a machine learning function designed for a variety of classification and regression tasks. The function accepts a data set, target variable, and a binary indicator of whether the target variable is continuous or not. The function cleans the data, encodes categorical features, selects the most important features, creates a stratified subset, trains a set of candidate models and tunes their hyper-parameters. Finally, the function retrains the best model on the whole data and returns the best model and the list of selected features.

Installation

This function requires Python 3.7 or later and the following dependencies:

    scikit-learn
    numpy
    pandas
    scipy

To install the required packages, you can use pip:

    pip install scikit-learn numpy pandas scipy

Usage:

   from theOneMLFunction import theOneMLFunction

    # data: pandas.DataFrame containing input data
    # targetVar: str containing the name of the target variable in data
    # isContinuous: bool indicating whether the problem is a regression problem (True) or a classification problem (False)
    # randomState: int setting the seed for the random number generator for reproducibility purposes

    bestModel, selectedFeatures = theOneMLFunction(data, targetVar, isContinuous, randomState)

The function returns the best trained model and the selected features, which are the 5 most important features selected using a feature selection method.
