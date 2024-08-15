from typing import Optional
import polars as pl
import seaborn as sns

# avoid the use of constant literals in our code and use named constants instead
COL_PASSENGER_ID = "PassengerId"
COL_SURVIVED = "Survived"
COL_PCLASS = "Pclass"
COL_NAME = "Name"
COL_SEX = "Sex"
COL_AGE = "Age"
COL_SIBSP = "SibSp"
COL_PARCH = "Parch"
COL_TICKET = "Ticket"
COL_FARE = "Fare"
COL_CABIN = "Cabin"
COL_EMBARKED = "Embarked"
COLS_USED_BY_MODEL = []

class Dataset:
    """explicitly represent the parameters that determine the data"""
    def __init__(self, drop_na: bool=False, num_samples: Optional[int]=None) -> None:
        self.drop_na = drop_na
        self.num_samples = num_samples

    def load_data(self):
        df = pl.read_csv("abc_titanic/data/raw.csv")
        if self.drop_na:
            return df.drop_nulls()
        if self.num_samples is not None:
            return df.sample(self.num_samples, seed=42)
        return df


if __name__ == "__main__":
    dataset = Dataset()
    raw_data = dataset.load_data()
    print(raw_data)

    sns.countplot(
        raw_data.to_pandas(),
        x="Survived"
    )
