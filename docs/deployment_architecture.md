# Arquitetura de Deploy

## Estratégia escolhida

Inferência **real-time** via FastAPI.

Essa escolha faz sentido porque o trabalho exige API de inferência e o cenário de churn pode apoiar ações imediatas de retenção no momento do atendimento ou da montagem de campanhas.

## Fluxo lógico

1. um cliente ou sistema interno envia um payload para `POST /predict`
2. a API valida o schema com Pydantic
3. o preprocessor salvo em `models/preprocessor.joblib` transforma os dados
4. a MLP carregada de `models/mlp_bundle.pt` calcula o score
5. a API devolve probabilidade, classe prevista, threshold e versão do modelo
6. logs estruturados registram latência e status da requisição

## Componentes

- FastAPI para serving
- artefatos locais versionados em `models/`
- MLflow para rastreamento de experimentos
- logs estruturados para observabilidade
- testes para schema, treino e API

## Evolução para produção

- containerizar a aplicação
- publicar em Render, Railway, Azure App Service, AWS ECS ou equivalente
- armazenar artefatos em repositório central
- proteger a API com autenticação
- separar ambiente de treino e ambiente de serving

## O que monitorar em produção

- latência p95
- disponibilidade do endpoint
- taxa de erro
- falhas de schema
- drift de features e score
- custo esperado da operação
