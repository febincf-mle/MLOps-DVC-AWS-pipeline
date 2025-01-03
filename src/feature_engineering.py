import os
from abc import abstractmethod, ABC

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import get_logger_instance, params_loader



# create logger instance
logger = get_logger_instance("feature_engineering", "INFO")


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def fit_data(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def vectorize(self, data: pd.DataFrame):
        pass



class TfIdfVectorizerStrategy(FeatureEngineeringStrategy):

    def __init__(self, max_features=5000):
        self._vectorizer = TfidfVectorizer(max_features=max_features, )

    def fit_data(self, data: pd.DataFrame):
        self._vectorizer.fit(data['tags'].values)
        logger.info("Fitting data completed")
        
    def vectorize(self, data: pd.DataFrame) -> pd.DataFrame:
        vectorized_tags = self._vectorizer.transform(data['tags'].values).toarray()
        vectorized_df = pd.DataFrame(vectorized_tags)
        vectorized_df['spam'] = data['spam']

        logger.info("Data vectorization successful")

        return vectorized_df
    

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self._strategy = strategy

    def build_vectorizer(self, data: pd.DataFrame) -> FeatureEngineeringStrategy:
        self._strategy.fit_data(data)
        return self._strategy
    


def main():

    # load csv files using pandas
    train_df = pd.read_csv("data/processed/processed_train_df.csv")
    test_df = pd.read_csv("data/processed/processed_test_df.csv")

    # load the parameters from the configuration file
    params = params_loader.params

    # create a feature engineer instance
    feature_engineer = FeatureEngineer(
        TfIdfVectorizerStrategy(max_features=params['feature_engineering']['max_features'])
        )
    
    vectorizer = feature_engineer.build_vectorizer(train_df)

    vectorized_train_df = vectorizer.vectorize(train_df)
    vectorized_test_df = vectorizer.vectorize(test_df)

    # create a directory for the final processed data
    os.makedirs("data/final", exist_ok=True)

    # create output files
    vectorized_train_df.to_csv("data/final/vectorized_train_df.csv", index=None)
    vectorized_test_df.to_csv("data/final/vectorized_test_df.csv", index=None)

    logger.info("Output Vectorized data in csv format success")



if __name__ == "__main__":
    main()