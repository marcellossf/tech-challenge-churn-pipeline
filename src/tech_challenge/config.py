from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


@dataclass(frozen=True)
class Settings:
    data_filename: str = field(
        default_factory=lambda: os.getenv("DATA_FILENAME", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    )
    target_column: str = field(default_factory=lambda: os.getenv("TARGET_COLUMN", "Churn"))
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "tech-challenge-churn")
    )
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))
    test_size: float = field(default_factory=lambda: float(os.getenv("TEST_SIZE", "0.2")))
    val_size: float = field(default_factory=lambda: float(os.getenv("VAL_SIZE", "0.2")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.001"))
    )
    max_epochs: int = field(default_factory=lambda: int(os.getenv("MAX_EPOCHS", "100")))
    patience: int = field(
        default_factory=lambda: int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))
    )
    mlp_hidden_dims: tuple[int, int] = (64, 32)
    mlp_dropout: float = field(default_factory=lambda: float(os.getenv("MLP_DROPOUT", "0.2")))
    decision_threshold: float = field(
        default_factory=lambda: float(os.getenv("DECISION_THRESHOLD", "0.35"))
    )
    false_positive_cost: float = field(
        default_factory=lambda: float(os.getenv("FALSE_POSITIVE_COST", "20"))
    )
    false_negative_cost: float = field(
        default_factory=lambda: float(os.getenv("FALSE_NEGATIVE_COST", "200"))
    )
    raw_data_candidates: tuple[str, ...] = (
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
    )

    @property
    def raw_data_dir(self) -> Path:
        return RAW_DATA_DIR

    @property
    def processed_data_dir(self) -> Path:
        return PROCESSED_DATA_DIR

    @property
    def models_dir(self) -> Path:
        return MODELS_DIR

    @property
    def reports_dir(self) -> Path:
        return REPORTS_DIR

    @property
    def mlruns_dir(self) -> Path:
        return ROOT_DIR / "mlruns"

    @property
    def mlflow_tracking_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI", self.mlruns_dir.as_uri())

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir / self.data_filename

    @property
    def preprocessor_path(self) -> Path:
        return self.models_dir / "preprocessor.joblib"

    @property
    def baseline_bundle_path(self) -> Path:
        return self.models_dir / "baseline_bundle.joblib"

    @property
    def mlp_bundle_path(self) -> Path:
        return self.models_dir / "mlp_bundle.pt"
