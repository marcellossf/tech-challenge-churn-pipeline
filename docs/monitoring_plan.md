# Plano de Monitoramento

## Monitoramento técnico

- disponibilidade do `GET /health`
- latência p95 do `POST /predict`
- taxa de erro 4xx e 5xx
- tempo de carregamento dos artefatos

### Alertas técnicos sugeridos

- latência p95 acima de 1 segundo por 15 minutos
- taxa de erro acima de 2%
- `model_ready = false` no `/health`

## Monitoramento de dados

- falhas de schema
- percentual de missing em `TotalCharges`
- mudança de distribuição em `MonthlyCharges`, `tenure` e `TotalCharges`
- mudança no mix de categorias de `Contract`, `InternetService` e `PaymentMethod`

### Alertas de dados sugeridos

- falha de schema acima de 1% das requisições
- aumento abrupto de missing
- desvio forte na distribuição das features principais

## Monitoramento do modelo

- taxa prevista de churn
- distribuição do score
- PR-AUC em revalidação offline
- custo esperado por campanha
- comportamento do threshold operacional `0.25`

### Alertas de modelo sugeridos

- PR-AUC offline abaixo de 0,60
- custo esperado crescendo de forma consistente
- taxa prevista de churn mudando mais de 5 pontos percentuais sem explicação de negócio

## Playbook de resposta

1. confirmar se o problema é técnico, de dados ou de comportamento do modelo
2. verificar logs, payloads inválidos e integridade dos artefatos
3. comparar distribuição atual com a base de treino
4. se necessário, reverter versão do modelo ou subir threshold conservador
5. abrir ciclo de retreino e nova validação antes de recolocar em produção
