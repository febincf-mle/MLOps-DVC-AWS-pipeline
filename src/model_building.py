import os
import pickle
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import ClassifierMixin

from utils import get_logger_instance, params_loader



# create a logger instance
logger = get_logger_instance("model_building", "INFO")


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> ClassifierMixin:
        pass



class MultinomialNBStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> ClassifierMixin:

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X is not a valid pandas dataframe")
        
        if not isinstance(y, pd.Series):
            raise TypeError("y is not a valid pandas series")
        
        clf_model = MultinomialNB(alpha=params['alpha'], fit_prior=params['fit_prior'])
        clf_model.fit(X, y)

        return clf_model
    


class ModelBuilder:

    def __init__(self, strategy: ModelBuildingStrategy):
        self._strategy = strategy


    def build_model(self, X: pd.DataFrame, y: pd.Series, params: dict) -> ClassifierMixin:
        return self._strategy.build_and_train_model(X, y, params)
    


def main():

    # load the data for training models
    train_df = pd.read_csv("data/final/vectorized_train_df.csv")

    # extract out the input and output cols
    target_col = 'spam'
    X_train = train_df[train_df.columns[:-1]]
    y_train = train_df[target_col]

    # load the parameters for the model
    params = params_loader.params

    # build and train the model
    model_builder = ModelBuilder(MultinomialNBStrategy())
    clf = model_builder.build_model(X_train, y_train, params=params['model_building'])

    # save the model to a file
    model_filename = "models/multinomialNB.pkl"
    logger.info("exporting the model using pickle")

    with open(model_filename, 'wb') as model_file:
        pickle.dump(clf, model_file)

    logger.info("model finished exporting")



if __name__== "__main__":
    main()