import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model(train: pd.DataFrame, model: Pipeline, cfg):
    TARGET = cfg.data.target
    x_train, x_test, y_train, y_test = train_test_split(train.drop(columns=[TARGET], axis=1),
                                                        train[TARGET],
                                                        random_state=cfg.data.random_state, shuffle=True)

    mlflow.set_experiment(cfg.experiments.name_experiment)
    with mlflow.start_run(run_name=cfg.experiments.run_name) as run:
        model.fit(x_train, y_train)
        y_test_preds = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_test_preds)
        mlflow.log_metric(key="mean_absolute_error_experiment_score", value=mae)

    return model
