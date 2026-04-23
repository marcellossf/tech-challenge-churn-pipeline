# Roteiro STAR

## Situation — 45s

Neste projeto, o problema de negócio é churn em uma operadora de telecom. A empresa quer identificar clientes com maior risco de cancelamento para priorizar retenção. O dataset escolhido foi o Telco Customer Churn, com 7.043 registros e variáveis tabulares de perfil, contrato, cobrança e serviços.

## Task — 40s

O objetivo técnico foi construir um pipeline profissional ponta a ponta. Isso incluiu EDA, ML Canvas, baselines com Scikit-Learn, uma MLP em PyTorch, tracking com MLflow, API FastAPI, testes automatizados, logging estruturado e documentação final com Model Card e plano de monitoramento.

## Action — 2min30s

Primeiro, fiz a exploração dos dados e identifiquei alguns sinais fortes de churn: contratos `Month-to-month`, clientes `Fiber optic` e pagamento por `Electronic check`. Também tratei `TotalCharges`, que vinha como texto, e removi `customerID` da modelagem.

Depois, construí baselines com `DummyClassifier`, `LogisticRegression` e `RandomForestClassifier`, usando pipeline reprodutível com imputação, padronização, one-hot encoding, seed fixa e validação cruzada estratificada. Todos os experimentos foram rastreados com MLflow.

Na sequência, implementei a MLP em PyTorch com duas camadas ocultas, ReLU, dropout, batching, early stopping e `BCEWithLogitsLoss` com ponderação para a classe positiva. Além das métricas padrão, fiz análise de custo por threshold com a hipótese de que falso negativo custa mais do que falso positivo.

Por fim, integrei o modelo à API FastAPI com endpoints `/health` e `/predict`, validação com Pydantic, logs estruturados e testes de smoke, schema e API.

## Result — 1min05s

O melhor baseline foi a Regressão Logística, com PR-AUC de 0,6334 e ROC-AUC de 0,8419. A MLP superou levemente esses valores, chegando a PR-AUC de 0,6348 e ROC-AUC de 0,8429. Mais importante, a MLP teve o menor custo esperado quando o threshold foi otimizado.

No threshold de custo ótimo, que ficou em `0.25`, a MLP alcançou custo esperado de `13.780`, melhor do que a Regressão Logística, que ficou em `14.380`. Por isso, a MLP foi escolhida como modelo final.

As principais limitações são o custo hipotético usado na análise, a ausência de deploy em nuvem até agora e a necessidade de monitorar drift em produção. Como próximos passos, eu publicaria a API e adicionaria revalidação periódica com dados mais recentes.
