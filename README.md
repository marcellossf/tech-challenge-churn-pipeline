# Tech Challenge Fase 01

Projeto para previsão de churn em telecom com pipeline de machine learning ponta a ponta.

## Objetivo

Desenvolver uma solução de classificação binária para identificar clientes com maior risco de churn, apoiando ações de retenção e priorização comercial.

## Escopo do projeto

O projeto foi estruturado para contemplar:

- análise exploratória de dados
- baselines em Scikit-Learn
- rede neural MLP em PyTorch
- rastreamento de experimentos com MLflow
- API de inferência com FastAPI
- testes automatizados
- documentação técnica de apoio

## Dataset

Dataset planejado para o projeto: **Telco Customer Churn**.

Arquivo esperado:

- `data/raw/Telco-Customer-Churn.csv`

## Estrutura do repositório

```text
data/
  raw/
  processed/
docs/
models/
notebooks/
reports/
src/tech_challenge/
tests/
```

## Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Comandos principais

```bash
make lint
make test
make eda
make train-baselines
make train-mlp
make api
make mlflow
```

## API planejada

Endpoints previstos:

- `GET /health`
- `POST /predict`

## Tecnologias

- Python
- Pandas e NumPy
- Scikit-Learn
- PyTorch
- MLflow
- FastAPI
- Pytest
- Ruff
