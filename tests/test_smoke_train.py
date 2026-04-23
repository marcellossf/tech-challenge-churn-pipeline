from __future__ import annotations

from pathlib import Path

from tech_challenge.config import Settings
from tech_challenge.models.train_baselines import train_baselines


def test_baseline_training_smoke(monkeypatch, telco_dataframe, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "telco.csv"
    telco_dataframe.to_csv(csv_path, index=False)

    monkeypatch.setenv("DATA_FILENAME", "telco.csv")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    from tech_challenge import config as config_module

    monkeypatch.setattr(config_module, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(config_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config_module, "REPORTS_DIR", tmp_path / "reports")

    results = train_baselines(Settings())
    assert not results.empty
    assert "pr_auc" in results.columns
