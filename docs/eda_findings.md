# EDA Findings

## Resumo do dataset

- CSV original com 7.043 linhas e 21 colunas
- Após remover `customerID` para modelagem, o dataframe analítico fica com 20 colunas
- Classe positiva `Churn = Yes`: 26,54%
- `TotalCharges` chega como texto e tem 11 valores ausentes após coerção numérica
- Dataset bruto sem duplicados exatos
- 22 perfis repetidos aparecem apenas após remover `customerID`

## Insights de negócio

- `Month-to-month` tem churn de 42,71%, muito acima de `One year` e `Two year`
- `Fiber optic` apresenta churn de 41,89%, acima de DSL e clientes sem internet
- `Electronic check` concentra churn de 45,29%, sugerindo uma boa alavanca para priorização

## Artefatos gerados

- `reports/eda_summary.json`
- `reports/eda_summary.md`
- `reports/eda_numeric_distributions.png`
- `reports/eda_contract_churn.png`
- `reports/eda_payment_churn.png`
- `reports/eda_internet_churn.png`
