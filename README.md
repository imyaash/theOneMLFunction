# theOneMLFunction
This Python script provides a one-stop machine learning function that can handle both regression and classification tasks. It includes data cleaning, feature selection, model training, hyperparameter tuning, and result printing.

This Python module includes theOneMLFunction, which is a machine learning function designed for a variety of classification and regression tasks. The function accepts a data set, target variable, and a binary indicator of whether the target variable is continuous or not. The function cleans the data, encodes categorical features, selects the most important features, creates a stratified subset, trains a set of candidate models and tunes their hyper-parameters. Finally, the function retrains the best model on the whole data and returns the best model and the list of selected features.

It requires the following dependencies to be installed: NumPy, Pandas, Scikit-learn, SciPy, and Random.

theOneMLFunction takes as input the following parameters:
  
    data: pandas.DataFrame, the input data.
    targetVar: str, the name of the target variable.
    isContinuous: bool, a binary indicator whether the target variable is continuous or not.
    randomState: int, the seed value for the random number generator.

The function returns the following:

    bestModel: scikit-learn model, the best trained model.
    selectedFeatures: list, the most important features.
