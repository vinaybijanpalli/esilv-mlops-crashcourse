from typing import List

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

CATEGORICAL_COLS = ["PULocationID", "DOLocationID", "passenger_count"]


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df[categorical_cols] = df[categorical_cols].fillna(-1).astype("int")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df
