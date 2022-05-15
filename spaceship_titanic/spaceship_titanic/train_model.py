# Execute this script to train the model

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from dagshub import DAGsHubLogger

from config import PATH_DATA_FOLDER, PATH_MODEL_FOLDER, PATH_LOGS_FOLDER, \
    COLUMN_ONEHOT


if __name__ == "__main__":
    logger = DAGsHubLogger(
        hparams_path=PATH_LOGS_FOLDER + "params.yml",
        should_log_metrics=False
    )

    train_data = pd.read_csv(PATH_DATA_FOLDER + "train.csv")
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    preprocessor = ColumnTransformer([
        ("one-hot", OneHotEncoder(handle_unknown="ignore"), COLUMN_ONEHOT),
    ])
    logger.log_hyperparams(preprocess_1="OneHotEncoder")
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(random_state=123))
    ])
    logger.log_hyperparams(model="Random Forest")
    pipeline.fit(X_train, y_train)
    print(f"Training score: {pipeline.score(X_train, y_train)}")

    joblib.dump(pipeline, PATH_MODEL_FOLDER + "pipeline.pkl")
    logger.save()
    logger.close()
