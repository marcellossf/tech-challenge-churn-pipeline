from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(dataframe: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    feature_frame = dataframe.drop(columns=[target_column])
    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in feature_frame.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: Iterable[str], categorical_features: Iterable[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ]
    )
