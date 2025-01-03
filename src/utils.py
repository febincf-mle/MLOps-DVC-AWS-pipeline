import os
import yaml
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score


BASE_DIR = os.path.realpath(".")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")


os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def get_logger_instance(name, level):

    # create a logger instance
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")

    # create filehandler
    filehandler = logging.FileHandler(os.path.join(LOGS_DIR, "ingest_data.log"))
    filehandler.setLevel(level)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    # create StreamHandler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def classification_report(y, y_pred):

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


class ParamsLoader:
    """
    A class to handle reading and managing configurations from a YAML file.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.params = {}
        self._load_params()

    def _load_params(self):
        """
        Loads parameters from the YAML file into the `params` attribute.
        Includes error handling for missing file, parsing errors, and invalid YAML structure.
        """
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"The file '{self.filepath}' does not exist.")

            with open(self.filepath, 'r') as file:
                self.params = yaml.safe_load(file)

                if not isinstance(self.params, dict):
                    raise ValueError("The YAML file does not contain a valid dictionary structure.")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise
        except ValueError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


params_loader = ParamsLoader("params.yaml")