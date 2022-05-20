from split_train import read_original_data

import pandas as pd


def test_read_original_data_returns_dataframe():
    df = read_original_data()
    assert isinstance(df, pd.DataFrame)
