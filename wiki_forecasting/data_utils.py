import os
import pandas as pd
import numpy as np
import holidays
import re
from datetime import timedelta

us_holidays = holidays.UnitedStates()  # or holidays.US()
project_root = "/home/joe/kaggle/wiki_forecasting"


def find_language(url):
    res = re.search('[a-z][a-z].wikipedia.org', url)
    if res:
        return res.group(0)[0:2]
    return 'na'


def get_features(df):
    features = dict()
    features["page"] = range(df.shape[0])
    features["ewm"] = df.ewm(halflife=30, axis=1).mean().iloc[:, -1]
    features["med1"] = df.iloc[:, -7:].median(axis=1)
    features["med2"] = df.iloc[:, -30:].median(axis=1)
    features["med3"] = df.iloc[:, -90:].median(axis=1)
    features["med_day"] = df.iloc[:, ::-7].median(axis=1)
    features["median"] = df.median(axis=1)
    features = pd.DataFrame(features)
    return pd.DataFrame(features)


def get_date_features(date):
    features = dict()
    features["day"] = date.dayofweek
    features["us_holiday"] = date in us_holidays
    features["date"] = date
    return features


def prepare_data(prediction_gap):
    df = pd.read_csv(os.path.join(project_root, "data/train_1.csv"),
                     index_col=0, nrows=1000).astype(np.float32)
    languages = pd.Categorical(df.index.map(find_language)).codes

    X = []
    y = []
    sums = []
    for i, date in enumerate(df.columns[-50:]):
        y.append(df[date])
        a = df.loc[:, :date].iloc[:, :-prediction_gap]
        date = pd.to_datetime(date)

        features = get_features(a, date)
        date_features = get_date_features(date)
        for name, val in date_features.items():
            features[name] = val
        features["language"] = languages
        features["date_code"] = i
        X.append(features)
    X = pd.concat(X)
    y = pd.concat(y)
    sums = pd.concat(sums)

    X = X[y.notnull()]
    sums = sums[y.notnull()]
    y = y[y.notnull()]

    train = X["date_code"] < 40
    return X.loc[train], y.loc[train], sums.loc[train], X.loc[~train], y.loc[~train], sums.loc[~train], df


def prepare_future_data(longest_gap):
    df = pd.read_csv(os.path.join(project_root, "data/train_1.csv"),
                     index_col=0).astype(np.float32)
    languages = pd.Categorical(df.index.map(find_language)).codes
    last_date = pd.to_datetime(df.columns[-1])
    features = get_features(df)
    for prediction_gap in range(1, longest_gap+1):
        date = last_date + timedelta(days=prediction_gap)
        date_features = get_date_features(date)
        for name, val in date_features.items():
            features[name] = val
        features["language"] = languages
        features["date_code"] = np.nan
        yield pd.DataFrame(features)