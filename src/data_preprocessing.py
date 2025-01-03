import os
import re

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

from utils import get_logger_instance


import nltk
nltk.download('stopwords')


# create logger instance
logger = get_logger_instance("data_preprocessing", "INFO")


# concrete class for email preprocessing
class EmailPreProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        """
        Constructor for EmailPreProcessor, takes in no arguments.

        Initializes the necessary components for email preprocessing:
        - stopwords: A set of common English stopwords.
        - stemmer: The Porter Stemmer for stemming words.
        - vectorizer: A TF-IDF vectorizer to convert text to numerical features.
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()


    def fit(self, X, y=None):
        """
        The fit method is responsible for learning the vocabulary and the TF-IDF scores
        from the email dataset. It also performs basic text preprocessing steps like:
        - Cleaning email content (removes HTML tags, special characters, and punctuation).
        - Removing stopwords.
        - Applying stemming.

        Parameters:
        - X: pandas DataFrame containing email content in the 'content' column.
        - y: Optional; target labels (not used here).

        Returns:
        - self: The fitted EmailPreProcessor object.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas dataframe")
        
        if 'content' not in X.columns:
            raise KeyError("No column or feature named content in X")

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        The transform method preprocesses the email data by applying the learned
        transformations on the raw email content.

        Preprocessing steps:
        - Removing HTML tags, special characters, punctuation.
        - Lowercasing the text.
        - Removing stopwords.
        - Stemming the tokens.

        Parameters:
        - X: pandas DataFrame containing the email content in the 'content' column.
        - y: Optional; target labels (not used here).

        Returns:
        - A pandas DataFrame with tags and spam as features.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas dataframe")
        
        if 'content' not in X.columns:
            raise KeyError("No column or feature named content in X")

        email_df = X.copy(deep=True)

        email_df['content'] = self.clean_email_content(email_df)
        email_df['tags'] = self.stopwords_removal(email_df)
        email_df['tags'] = self.stem_conversion(email_df)

        preprocessed_df = pd.DataFrame()
        preprocessed_df['tags'] = email_df['tags']
        preprocessed_df['spam'] = email_df['spam']

        return preprocessed_df


    def clean_email_content(self, email_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes html tags, unwanted special characters and punctuations and extra spaces
        from email content.

        paramters: email_df -> pandas Dataframe
        """
        def clean_text(text: str) -> str:
            # remove HTML tags using BeautifulSoup
            text = BeautifulSoup(text, "html.parser").get_text()

            # remove non-ASCII characters and unwanted symbols (e.g., "\x01")
            text = re.sub(r'[^\x00-\x7F]+', '', text)

            # remove all punctuation using regex (except spaces)
            text = re.sub(r'[^\w\s]', '', text)

            # normalize multiple spaces or newlines to a single space
            text = re.sub(r'\s+', ' ', text).strip()

            # convert the text string to lower case for noramlization
            text = text.lower()
            return text

        logger.info("text cleaning about to start")
        return email_df['content'].apply(clean_text)


    def stopwords_removal(self, email_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes stopwords from the email text.

        Parameters:
        - email_df: pandas DataFrame containing the 'content' column with tokenized text.

        Returns:
        - A pandas Series with the tokenized text after removing stopwords.
        """
        def stopwords_removal_helper(text):
            words = text.split()
            return [word for word in words if word not in self.stop_words]

        logger.info("stopwords removal beginning")
        return email_df['content'].apply(stopwords_removal_helper)


    def stem_conversion(self, email_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies stemming to the tokens in the email content.

        Parameters:
        - email_df: pandas DataFrame containing tokenized email content (tags).

        Returns:
        - A pandas Series with the stemmed words.
        """
        def stem_conversion_helper(words):
            # join the list values as a string to pass it to TF-IDF Vectorizer
            return " ".join([self.stemmer.stem(word) for word in words])

        logger.info("stemming words beginning")
        return email_df['tags'].apply(stem_conversion_helper)
    


def main():

    # load the train and test data
    train_df = pd.read_csv("data/split/train.csv")
    test_df = pd.read_csv("data/split/test.csv")

    # create an instance of the EmailPreProcessor to preprocess the data
    preprocessor = EmailPreProcessor()
    preprocessor.fit(train_df)

    transformed_train_df = preprocessor.transform(train_df)
    transformed_test_df = preprocessor.transform(test_df)

    # create the processed data directory
    os.makedirs("data/processed", exist_ok=True)

    # create output files
    transformed_train_df.to_csv("data/processed/processed_train_df.csv", index=None)
    transformed_test_df.to_csv("data/processed/processed_test_df.csv", index=None)

    logger.info("Preprocessing step completed.")


if __name__ == "__main__":
    main()