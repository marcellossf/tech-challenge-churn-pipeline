# Roteiro STAR para o video de 5 minutos

Este roteiro foi pensado para ser falado de forma natural, mostrando evidencias reais do projeto na tela. A ideia nao e ler palavra por palavra, e sim usar como guia para manter ritmo, cobrir todos os pontos avaliados e nao passar de 5 minutos.

## Preparacao antes de gravar

Deixe abertas as seguintes telas:

- GitHub do projeto na pagina principal do repositorio.
- README com a tabela de resultados.
- `notebooks/01_eda_baselines.ipynb` ou os graficos em `reports/`.
- `reports/model_comparison.md`.
- `src/tech_challenge/models/train_mlp.py`.
- `src/tech_challenge/api/app.py`.
- Terminal com `python -m pytest` ja executado ou pronto para executar.
- Opcional: MLflow UI, se estiver rapido de abrir.

Comandos uteis para mostrar na gravacao:

```powershell
python -m ruff check src tests
python -m pytest
python -m uvicorn tech_challenge.api.app:app --reload
```

## 0:00 a 0:15 - Abertura

Fala sugerida:

> Ola, eu sou o Marcello e este e meu projeto do Tech Challenge Fase 01. Eu construi um pipeline end-to-end para previsao de churn em telecom, usando EDA, baselines, uma rede neural em PyTorch, MLflow, FastAPI, testes automatizados e documentacao tecnica.

Tela:

- Mostre o GitHub do projeto.
- Aponte rapidamente para a estrutura de pastas.

## 0:15 a 1:00 - Situation

Fala sugerida:

> A situacao de negocio e o problema de churn em uma operadora de telecom. A empresa quer identificar clientes com maior risco de cancelamento para priorizar acoes de retencao. O dataset escolhido foi o Telco Customer Churn, com 7.043 registros e 21 colunas na base bruta. A variavel alvo e `Churn`.

> A decisao de negocio importante foi tratar falso negativo como erro mais caro. Ou seja, deixar de identificar um cliente que realmente vai cancelar e pior do que abordar um cliente que talvez nao cancelasse.

Tela:

- Mostre o README na parte de objetivo e dataset.
- Se der tempo, mostre `docs/ml_canvas.md`.

## 1:00 a 1:35 - Task

Fala sugerida:

> A tarefa foi criar uma solucao reproduzivel e profissional. Isso envolveu preparar os dados, fazer EDA, criar baselines, treinar uma MLP em PyTorch, rastrear experimentos com MLflow, expor uma API de inferencia, escrever testes e documentar limitacoes, arquitetura e monitoramento.

> A metrica principal escolhida foi PR-AUC, porque o churn e um problema em que a classe positiva e especialmente importante. Tambem acompanhei ROC-AUC, F1, precision, recall, accuracy e custo esperado por threshold.

Tela:

- Mostre README na parte de pipeline implementado.
- Mostre a estrutura `src/`, `tests/`, `docs/` e `reports/`.

## 1:35 a 3:45 - Action

Fala sugerida:

> Na primeira etapa, fiz a EDA. Removi `customerID` da modelagem, tratei `TotalCharges`, que vinha como texto, analisei balanceamento da target e observei padroes de churn por contrato, servico de internet e forma de pagamento. Alguns sinais fortes foram clientes com contrato `Month-to-month`, internet `Fiber optic` e pagamento por `Electronic check`.

Tela:

- Mostre graficos em `reports/` ou o notebook `01_eda_baselines.ipynb`.

Fala sugerida:

> Depois criei baselines com Scikit-Learn: `DummyClassifier`, `LogisticRegression` e `RandomForestClassifier`. O preprocessamento ficou dentro de um pipeline reproduzivel com imputacao, padronizacao de variaveis numericas, one-hot encoding para categoricas, seed fixa e validacao cruzada estratificada. Os experimentos foram registrados com MLflow.

Tela:

- Mostre `src/tech_challenge/models/train_baselines.py`.
- Mostre `reports/baseline_results.md`.

Fala sugerida:

> Em seguida implementei a MLP em PyTorch. A arquitetura tem duas camadas ocultas, 64 e 32 neuronios, ativacao ReLU, dropout de 0.2 e saida binaria com `BCEWithLogitsLoss`. O treino usa batching, early stopping e ponderacao da classe positiva para lidar com o desbalanceamento.

Tela:

- Mostre `src/tech_challenge/models/mlp.py`.
- Mostre `src/tech_challenge/models/train_mlp.py`.
- Se quiser, mostre `reports/mlp_artifacts/training_history.png`.

Fala sugerida:

> Um ponto importante foi que eu nao escolhi o modelo apenas pela acuracia. Fiz uma analise de custo assumindo falso positivo com custo 20 e falso negativo com custo 200. Com isso, varri varios thresholds e escolhi o ponto de operacao que minimizava o custo esperado.

Tela:

- Mostre `docs/business_cost_analysis.md`.
- Mostre `reports/threshold_analysis.json` ou `reports/mlp_results.md`.

Fala sugerida:

> Por fim, empacotei o modelo para inferencia com FastAPI. A API tem `/health` e `/predict`, valida entrada com Pydantic, carrega o mesmo preprocessador usado no treino e retorna probabilidade, classe prevista, threshold e versao do modelo. Tambem inclui logging estruturado com request id e latencia.

Tela:

- Mostre `src/tech_challenge/api/app.py`.
- Mostre `src/tech_challenge/api/service.py`.

## 3:45 a 4:40 - Result

Fala sugerida:

> Nos resultados, a regressao logistica foi um baseline muito forte, com PR-AUC de 0.6334 e ROC-AUC de 0.8419. A MLP ficou levemente acima, com PR-AUC de 0.6348 e ROC-AUC de 0.8429.

> A diferenca mais importante apareceu na analise de custo. A MLP teve melhor custo esperado quando o threshold foi ajustado para 0.25, chegando a 13.780. A regressao logistica ficou em 14.380 no melhor threshold. Por isso escolhi a MLP como modelo final, porque ela se alinhou melhor ao objetivo de reduzir falsos negativos e apoiar retencao.

Tela:

- Mostre `reports/model_comparison.md`.
- Mostre a tabela de resultados do README.

## 4:40 a 5:00 - Fechamento

Fala sugerida:

> Para fechar, o projeto tambem tem testes automatizados, Model Card, arquitetura de deploy e plano de monitoramento. As principais limitacoes sao que o custo usado e uma hipotese de negocio, nao houve validacao temporal e o modelo precisaria de monitoramento de drift em producao. Como proximo passo, eu publicaria a API em nuvem e criaria uma rotina periodica de revalidacao e retreino.

Tela:

- Rode ou mostre `python -m pytest`.
- Mostre `docs/model_card.md` e `docs/monitoring_plan.md`.

## Checklist do video

- Mostrar o repositorio no GitHub.
- Mostrar a estrutura do projeto.
- Explicar o problema de negocio e a target `Churn`.
- Citar PR-AUC como metrica principal.
- Mostrar EDA ou graficos.
- Mostrar baselines e MLflow.
- Mostrar a MLP em PyTorch.
- Explicar early stopping, batching e loss.
- Explicar threshold por custo.
- Mostrar API FastAPI.
- Mostrar testes passando.
- Fechar com limitacoes e proximos passos.

## Frase curta se o tempo estiver acabando

> Em resumo, o projeto entrega um fluxo completo: dados, experimentos, MLP, API, testes e documentacao. A MLP foi escolhida nao por acuracia, mas por melhor PR-AUC e menor custo esperado no threshold operacional de 0.25.
