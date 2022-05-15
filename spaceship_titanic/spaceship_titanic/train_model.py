# Execute this script to train the model

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from config import PATH_DATA_FOLDER, PATH_MODEL_FOLDER, COLUMN_ONEHOT


if __name__ == "__main__":
    train_data = pd.read_csv(PATH_DATA_FOLDER + "train.csv")
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    preprocessor = ColumnTransformer([
        ("one-hot", OneHotEncoder(handle_unknown="ignore"), COLUMN_ONEHOT),
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(random_state=123))
    ])
    pipeline.fit(X_train, y_train)
    print(f"Training score: {pipeline.score(X_train, y_train)}")

    joblib.dump(pipeline, PATH_MODEL_FOLDER + "pipeline.pkl")
