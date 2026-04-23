# Model Card

## Modelo

- Nome: `mlp_pytorch`
- Versão: `0.1.0`
- Tipo: rede neural MLP para classificação binária
- Framework: PyTorch
- Arquitetura: input -> 64 -> 32 -> 1
- Ativação: ReLU
- Regularização: Dropout 0,2
- Loss: `BCEWithLogitsLoss` com ponderação para classe positiva

## Dados

- Dataset: Telco Customer Churn
- Fonte: IBM / versão pública em GitHub
- Base bruta: 7.043 linhas e 21 colunas
- Base de modelagem: 20 colunas após remoção de `customerID`
- Target: `Churn`
- Distribuição da classe positiva: 26,54%
- Tratamentos principais:
  - remoção de `customerID`
  - coerção numérica de `TotalCharges`
  - imputação para numéricas e categóricas
  - padronização de variáveis numéricas
  - one-hot encoding para categóricas

## Uso pretendido

Priorizar clientes para campanhas de retenção, recomendação de contato comercial e apoio à operação de atendimento.

## Não usar para

- negar atendimento
- aplicar penalidade automática
- substituir decisão humana de retenção
- inferir causa exata do churn

## Métricas do modelo final

No threshold de avaliação `0.35`:

- PR-AUC: 0,6348
- ROC-AUC: 0,8429
- F1: 0,5986
- Precision: 0,4462
- Recall: 0,9091
- Accuracy: 0,6764
- Custo esperado: 15.240

No threshold selecionado por custo `0.25`:

- melhor custo esperado: 13.780
- F1: 0,5712
- Precision: 0,4075
- Recall: 0,9545

## Comparação com baselines

- A MLP teve o melhor PR-AUC e o menor custo esperado.
- A Regressão Logística teve F1, precision e accuracy mais altos no threshold de avaliação.
- A decisão final favoreceu a MLP por melhor alinhamento com o objetivo de retenção e menor custo total esperado.

## Limitações

- custo de negócio modelado com hipóteses simples
- ausência de validação temporal
- threshold ótimo pode variar por campanha ou segmento
- resultado depende da qualidade e estabilidade dos dados de entrada

## Vieses e cenários de falha

- mudança no mix de clientes pode gerar drift
- grupos pouco representados podem ter desempenho inferior
- novos planos, políticas comerciais ou métodos de pagamento podem alterar o padrão aprendido
- `TotalCharges` ausente ou mal preenchido pode afetar a inferência

## Monitoramento recomendado

- latência e taxa de erro da API
- falhas de schema
- taxa prevista de churn
- drift de features e do score
- revalidação periódica de PR-AUC e custo esperado
