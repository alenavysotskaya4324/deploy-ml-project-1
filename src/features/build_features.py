import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer


def create_date_features(data: pd.DataFrame) -> pd.DataFrame:
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.loc[:, 'day'] = data['datetime'].dt.day
        data.loc[:, 'month'] = data['datetime'].dt.month
        data.loc[:, 'year'] = data['datetime'].dt.year
        data.loc[:, 'hour'] = data['datetime'].dt.hour
        data.loc[:, 'dayofweek'] = data['datetime'].dt.dayofweek
        data.loc[:, 'weekend'] = np.where(data['dayofweek'].isin([5, 6]), 1, 0)

    return data


def create_weather_feature(data: pd.DataFrame) -> pd.DataFrame:
    if 'weather' in data.columns:
        data.loc[:, 'good_weather'] = data['weather'].map(lambda x: x == 1).astype(int)

    return data


def transform_wind_speed_feature(data: pd.DataFrame) -> pd.DataFrame:
    if 'windspeed' in data.columns:
        transformer = PowerTransformer()
        data['windspeed'] = transformer.fit_transform(data['windspeed'].values.reshape(-1, 1))

    return data


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = create_date_features(data)
    data = create_weather_feature(data)

    return data
