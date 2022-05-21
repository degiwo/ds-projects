# Split the data in train.csv to train and test data sets.

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from spaceship_titanic.config import (
    PATH_DATA_FOLDER,
    PATH_LOGS_FOLDER,
    FILE_NAME_ORIGINAL_DATA,
    FILE_NAME_TRAIN_DATA,
    FILE_NAME_TEST_DATA,
)


def read_original_data() -> pd.DataFrame:
    data_path = PATH_DATA_FOLDER + FILE_NAME_ORIGINAL_DATA
    try:
        df = pd.read_csv(data_path)
        logging.info("Original data successfully read")
        return df
    except FileNotFoundError:
        logging.error(
            f"Original data not found, please check if {data_path} exists"
        )


def split_and_save(data: pd.DataFrame) -> None:
    """
    Use to split the original data to train and test for building a model.
    """
    try:
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=0
        )
        train_data.to_csv(PATH_DATA_FOLDER + FILE_NAME_TRAIN_DATA)
        test_data.to_csv(PATH_DATA_FOLDER + FILE_NAME_TEST_DATA)
        logging.info("Train and test data successfully splitted")
    except TypeError:
        logging.error(
            f"Input data is {type(data)}, but DataFrame is needed to split"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename=PATH_LOGS_FOLDER + "split_train.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("split_train.py script started")
    original_data = read_original_data()
    split_and_save(original_data)
