from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch

from tech_challenge.api.schemas import PredictRequest
from tech_challenge.config import Settings
from tech_challenge.models.mlp import ChurnMLP


class PredictionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.preprocessor = joblib.load(settings.preprocessor_path)
        checkpoint = torch.load(settings.mlp_bundle_path, map_location="cpu")
        self.threshold = float(checkpoint["threshold"])
        self.model_version = str(checkpoint.get("model_version", "0.1.0"))
        self.model = ChurnMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dropout=float(checkpoint["dropout"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, request: PredictRequest) -> dict[str, float | int | str]:
        payload = {
            "gender": request.gender,
            "SeniorCitizen": request.senior_citizen,
            "Partner": request.partner,
            "Dependents": request.dependents,
            "tenure": request.tenure,
            "PhoneService": request.phone_service,
            "MultipleLines": request.multiple_lines,
            "InternetService": request.internet_service,
            "OnlineSecurity": request.online_security,
            "OnlineBackup": request.online_backup,
            "DeviceProtection": request.device_protection,
            "TechSupport": request.tech_support,
            "StreamingTV": request.streaming_tv,
            "StreamingMovies": request.streaming_movies,
            "Contract": request.contract,
            "PaperlessBilling": request.paperless_billing,
            "PaymentMethod": request.payment_method,
            "MonthlyCharges": request.monthly_charges,
            "TotalCharges": request.total_charges,
        }
        frame = pd.DataFrame([payload])
        transformed = self.preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        with torch.no_grad():
            score = torch.sigmoid(
                self.model(torch.tensor(np.asarray(transformed), dtype=torch.float32))
            ).item()
        return {
            "churn_probability": float(score),
            "churn_prediction": int(score >= self.threshold),
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService(Settings())


def model_ready(settings: Settings) -> bool:
    return settings.preprocessor_path.exists() and settings.mlp_bundle_path.exists()


def model_version(settings: Settings) -> str | None:
    if not settings.mlp_bundle_path.exists():
        return None
    checkpoint = torch.load(settings.mlp_bundle_path, map_location="cpu")
    return str(checkpoint.get("model_version", "0.1.0"))
