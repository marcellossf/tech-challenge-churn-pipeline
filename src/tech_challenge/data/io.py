from __future__ import annotations

from pathlib import Path

import pandas as pd

from tech_challenge.config import Settings
from tech_challenge.data.schema import build_input_schema


def resolve_raw_data_path(settings: Settings) -> Path:
    path = settings.raw_data_path
    if not path.exists():
        fallback_path = next(
            (
                settings.raw_data_dir / candidate
                for candidate in settings.raw_data_candidates
                if (settings.raw_data_dir / candidate).exists()
            ),
            None,
        )
        if fallback_path is None:
            raise FileNotFoundError(f"Dataset not found at {path}")
        path = fallback_path
    return path


def load_raw_dataset(settings: Settings) -> pd.DataFrame:
    return pd.read_csv(resolve_raw_data_path(settings))


def prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "customerID" in dataframe.columns:
        dataframe = dataframe.drop(columns=["customerID"])

    if "TotalCharges" in dataframe.columns:
        dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")

    return dataframe


def load_dataset(settings: Settings) -> pd.DataFrame:
    dataframe = prepare_dataset(load_raw_dataset(settings))
    dataframe = dataframe.copy()
    build_input_schema(settings.target_column).validate(dataframe, lazy=True)
    return dataframe


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
