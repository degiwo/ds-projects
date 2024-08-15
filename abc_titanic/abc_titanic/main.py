import polars as pl
import seaborn as sns

if __name__ == "__main__":
    raw_data = pl.read_csv("abc_titanic/data/raw.csv")

    sns.countplot(
        raw_data.to_pandas(),
        x="Survived"
    )
