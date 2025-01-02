import os
import logging
from abc import ABC, abstractmethod

import pandas as pd

from utils import get_logger_instance, DATA_DIR, BASE_DIR


# create a logger instance
logger = get_logger_instance("ingest_data", logging.INFO)


# Abstract class for ingesting data
class DataIngestor(ABC):
    @abstractmethod
    def ingest_data(self, filepath: str) -> pd.DataFrame:
        pass


# Concrete class for ingesting data (.csv files)
class CSVDataIngestor(DataIngestor):
    def ingest_data(self, filepath: str) -> pd.DataFrame:
        """
        Ingests csv file data.
        
        Parameters:
        filepath: the absolute path of the file.

        Returns:
        pd.DataFrame: pandas dataframe with the data ingested.
        """
        if os.path.isfile(filepath) and filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
            logger.info("Data ingestion completed.")
            return df
        
        else:
            logger.error("The csv file does not exist.")
            raise FileNotFoundError
    

# Implement a factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_ext: str) -> DataIngestor:
        """
        Get the appropriate DataIngestor based on the file format.
        
        Parameters:
        file_ext: The format/file extension.
        
        Reuturns:
        DataIngestor: The DataIngestor which can ingest the data from the given format."""
        if file_ext == ".csv":
            return CSVDataIngestor()
        else:
            raise ValueError("No ingestor found for this file extension")
        


def main():

    data_source = os.path.join(BASE_DIR, "experiments")
    files = os.listdir(data_source)

    csv_files = [ file for file in files if file.endswith(".csv") ]
    
    if len(csv_files) > 1:
        raise ValueError("Multiple csv files present in the Destination")
    
    filename = csv_files[0]
    filepath = os.path.join(data_source, filename)
    file_ext = os.path.splitext(filename)[-1]

    ingestor = DataIngestorFactory.get_data_ingestor(file_ext)
    df = ingestor.ingest_data(filepath)

    raw_data_path = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_data_path, exist_ok=True)

    output_filepath = os.path.join(raw_data_path, "raw-data.csv")
    df.to_csv(output_filepath, index=None)


if __name__ == "__main__":
    main()