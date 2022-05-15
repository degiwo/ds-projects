# Split the data in train.csv to train and test data sets.

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PATH_DATA_FOLDER


def read_original_data() -> pd.DataFrame:
    return pd.read_csv(PATH_DATA_FOLDER + "original_data.csv")


def split_and_save(data: pd.DataFrame) -> None:
    """
    Use to split the original data to train and test for building a model.
    """
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=0
    )
    train_data.to_csv(PATH_DATA_FOLDER + "train.csv")
    test_data.to_csv(PATH_DATA_FOLDER + "test.csv")


if __name__ == "__main__":
    original_data = read_original_data()
    split_and_save(original_data)
