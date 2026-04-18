PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest

eda:
	$(PYTHON) -m tech_challenge.data.eda

run:
	$(PYTHON) -m uvicorn tech_challenge.api.app:app --reload

api:
	$(PYTHON) -m uvicorn tech_challenge.api.app:app --host 0.0.0.0 --port 8000

train-baselines:
	$(PYTHON) -m tech_challenge.models.train_baselines

train-mlp:
	$(PYTHON) -m tech_challenge.models.train_mlp

mlflow:
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns
