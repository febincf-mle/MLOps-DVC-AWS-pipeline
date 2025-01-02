import os
import json
import pickle
import pandas as pd

from abc import ABC, abstractmethod
from utils import classification_report, get_logger_instance

from sklearn.base import ClassifierMixin


# create logger instance
logger = get_logger_instance("model_evaluation", "INFO")


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self):
        pass


class ClassificationEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self,
                        estimator: ClassifierMixin, 
                        X_train: pd.DataFrame, 
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series
                        ) -> dict:
        
        y_train_predictions = estimator.predict(X_train)
        y_test_predictions = estimator.predict(X_test)

        train_report = classification_report(y_train, y_train_predictions)
        test_report = classification_report(y_test, y_test_predictions)

        return {
            'train_report': train_report,
            'test_report': test_report
        }


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def evaluate(self,
                    estimator: ClassifierMixin, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series
                    ) -> dict:
        
        return self._strategy.evaluate_model(estimator,
                                        X_train,
                                        y_train,
                                        X_test,
                                        y_test,
                                        )
    

def main():

    # load csv files using pandas
    train_df = pd.read_csv('data/final/vectorized_train_df.csv')
    test_df = pd.read_csv('data/final/vectorized_test_df.csv')

    # separate the input and output columns
    target_col = 'spam'
    X_train = train_df[train_df.columns[:-1]]
    y_train = train_df[target_col]

    X_test = test_df[test_df.columns[:-1]]
    y_test = test_df[target_col]

    # load the model
    logger.info("loading the model from the pickle file")
    with open('models/multinomialNB.pkl', 'rb') as file:
        model = pickle.load(file)

    # evaluate the model
    logger.info("Started Model evaluation")
    model_evaluator = ModelEvaluator(ClassificationEvaluationStrategy())
    evaluation_report = model_evaluator.evaluate(model,
                              X_train,
                              y_train,
                              X_test,
                              y_test
                              )
    
    # write to a JSON file
    logger.info("started writing the evaluation report to a file")
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/evaluation_report.json', 'w') as file:
        json.dump(evaluation_report, file)

    logger.info("Evaluation report successfully written to the file")



if __name__ == "__main__":
    main()