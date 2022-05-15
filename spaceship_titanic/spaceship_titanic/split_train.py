# Split the data in train.csv to train and test data sets.

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    PATH_DATA_FOLDER,
    FILE_NAME_ORIGINAL_DATA,
    FILE_NAME_TRAIN_DATA,
    FILE_NAME_TEST_DATA,
)


def read_original_data() -> pd.DataFrame:
    return pd.read_csv(PATH_DATA_FOLDER + FILE_NAME_ORIGINAL_DATA)


def split_and_save(data: pd.DataFrame) -> None:
    """
    Use to split the original data to train and test for building a model.
    """
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=0
    )
    train_data.to_csv(PATH_DATA_FOLDER + FILE_NAME_TRAIN_DATA)
    test_data.to_csv(PATH_DATA_FOLDER + FILE_NAME_TEST_DATA)


if __name__ == "__main__":
    original_data = read_original_data()
    split_and_save(original_data)
