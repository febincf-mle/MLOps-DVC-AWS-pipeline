import os
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