# Análise de Custo

## Hipótese usada

- custo de falso positivo: R$ 20
- custo de falso negativo: R$ 200

Essa hipótese representa um cenário em que abordar um cliente sem necessidade é barato perto de perder um cliente que realmente iria cancelar.

## Objetivo

Escolher o threshold que reduz o custo total esperado, em vez de assumir automaticamente `0.5` como ponto de decisão.

## Resultado da MLP

- threshold de comparação padrão: `0.35`
- custo esperado em `0.35`: `15.240`
- melhor threshold por custo: `0.25`
- melhor custo esperado: `13.780`

Resumo da operação da MLP em `0.25`:

- `TP = 357`
- `FP = 519`
- `FN = 17`
- `TN = 516`

## Comparação com baselines

- melhor custo da MLP: `13.780`
- melhor custo da Regressão Logística: `14.380`
- melhor custo da Random Forest: `17.060`
- melhor custo do Dummy: `20.700`

## Interpretação

- A Regressão Logística é forte como baseline e teve F1 melhor no threshold de avaliação.
- A MLP foi escolhida porque, com ajuste de threshold, entregou o menor custo total esperado.
- O threshold ótimo não ficou em `0.5`; ele caiu para `0.25`, o que aumenta recall e reduz falsos negativos.

## Recomendação

Usar a MLP com threshold operacional inicial de `0.25` para campanhas de retenção, revisando esse valor quando o custo real de contato e perda de cliente estiver disponível.
