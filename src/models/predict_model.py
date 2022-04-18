import pandas as pd
from sklearn.pipeline import Pipeline
import global_config.config as config


def predict_model(test: pd.DataFrame, model: Pipeline):
    ss = pd.read_csv(config.DATA_PATH+config.SAMPLE_SUBMISSION)
    predicts = model.predict(test)
    ss['count'] = predicts.astype(int)
    ss.to_csv(config.DATA_PATH+config.SAMPLE_SUBMISSION_WITH_ANSWERS, index=False)
