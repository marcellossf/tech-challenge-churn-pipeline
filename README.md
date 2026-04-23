# Tech Challenge Fase 01

Pipeline end-to-end para predição de churn em telecom, cobrindo análise exploratória, baselines, rede neural MLP em PyTorch, rastreamento de experimentos com MLflow, API FastAPI, testes automatizados e documentação de apoio.

## Objetivo

O projeto busca identificar clientes com maior risco de cancelamento para apoiar ações de retenção. A hipótese de negócio adotada é que falso negativo custa mais do que falso positivo, então a análise final considera não só métricas tradicionais, mas também custo esperado por threshold.

## Dataset

- Base: Telco Customer Churn
- Arquivo esperado: `data/raw/Telco-Customer-Churn.csv`
- Tamanho da base: 7.043 registros e 21 colunas na versão bruta
- Target: `Churn`

## Estrutura do projeto

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

## Setup

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

## Pipeline implementado

1. EDA reproduzível com geração de artefatos em `reports/`
2. ML Canvas e documentação de contexto de negócio em `docs/`
3. Baselines com `DummyClassifier`, `LogisticRegression` e `RandomForestClassifier`
4. MLP em PyTorch com duas camadas ocultas, dropout, batching e early stopping
5. Seleção de threshold por custo esperado
6. API FastAPI com `GET /health` e `POST /predict`
7. Testes de smoke, schema e API

## Resultados principais

Comparação no conjunto de teste:

| Modelo | PR-AUC | ROC-AUC | F1 | Recall | Precision |
| --- | --- | --- | --- | --- | --- |
| MLP PyTorch | 0.6348 | 0.8429 | 0.5986 | 0.9091 | 0.4462 |
| Logistic Regression | 0.6334 | 0.8419 | 0.6147 | 0.7059 | 0.5443 |
| Random Forest | 0.6055 | 0.8171 | 0.5878 | 0.6578 | 0.5313 |
| Dummy | 0.2654 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

Resumo da decisão final:

- modelo final escolhido: `mlp_pytorch`
- threshold de avaliação: `0.35`
- threshold operacional selecionado por custo: `0.25`
- melhor custo esperado da MLP: `13.780`
- melhor custo esperado da regressão logística: `14.380`

## API

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "0.1.0"
}
```

### Predição

```powershell
$body = @{
  gender = "Female"
  SeniorCitizen = 0
  Partner = "Yes"
  Dependents = "No"
  tenure = 24
  PhoneService = "Yes"
  MultipleLines = "No"
  InternetService = "Fiber optic"
  OnlineSecurity = "No"
  OnlineBackup = "Yes"
  DeviceProtection = "No"
  TechSupport = "No"
  StreamingTV = "Yes"
  StreamingMovies = "Yes"
  Contract = "Month-to-month"
  PaperlessBilling = "Yes"
  PaymentMethod = "Electronic check"
  MonthlyCharges = 89.9
  TotalCharges = 2112.4
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

Resposta esperada:

```json
{
  "churn_probability": 0.7813,
  "churn_prediction": 1,
  "threshold": 0.25,
  "model_version": "0.1.0"
}
```

## Testes

Cobertura mínima implementada:

- `tests/test_smoke_train.py`: smoke test do treino baseline
- `tests/test_schema.py`: validação de schema com Pandera
- `tests/test_api.py`: validação dos endpoints `/health` e `/predict`

## Documentação

- `docs/ml_canvas.md`
- `docs/eda_findings.md`
- `docs/business_cost_analysis.md`
- `docs/model_card.md`
- `docs/deployment_architecture.md`
- `docs/monitoring_plan.md`
- `docs/video_star_script.md`

## Arquitetura escolhida

Deploy pensado para inferência real-time via FastAPI, com preprocessador salvo em `models/preprocessor.joblib` e modelo MLP salvo em `models/mlp_bundle.pt`. Em produção, a evolução natural é containerizar a aplicação, publicar a API e centralizar artefatos e logs.

## Limitações

- análise de custo baseada em hipótese simplificada de negócio
- ausência de validação temporal
- threshold ideal pode variar por campanha, segmento e política comercial
- monitoramento contínuo é necessário para detectar drift de dados e score
