import os
from typing import Tuple
from abc import abstractmethod, ABC

import pandas as pd
from sklearn.model_selection import train_test_split



class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, data: pd.DataFrame):
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def split_data(self, data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe")
        
        train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_df, test_df


class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy, test_size: float = 0.2, random_state: int = 10):
        self._test_size = test_size
        self._strategy = strategy
        self._random_state = random_state

    def set_strategy(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def split(self, data: pd.DataFrame):
        return self._strategy.split_data(data, self._test_size, self._random_state)
    


def main():

    raw_data = pd.read_csv("data/raw/raw-data.csv")

    data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(), 0.2, 44)
    train_df, test_df = data_splitter.split(raw_data)

    # create output files and directories
    train_df.to_csv("data/raw/train.csv", index=None)
    test_df.to_csv("data/raw/test.csv", index=None)



if __name__ == "__main__":
    main()