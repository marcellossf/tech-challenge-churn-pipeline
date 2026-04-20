# ML Canvas

## Problema de negócio

Antecipar clientes com alto risco de churn para priorizar ações de retenção em telecom.

## Stakeholders

- Diretoria de negócio
- CRM e retenção
- Operação de atendimento
- Time de dados

## Predição

- Variável alvo: `Churn`
- Tipo de problema: classificação binária
- Janela de uso: suporte a campanhas e decisões operacionais de retenção
- Dataset adotado: Telco Customer Churn com 7.043 registros e 21 colunas no CSV original

## Métricas

- Métrica técnica principal: PR-AUC
- Métricas secundárias: ROC-AUC, F1, Recall, Precision
- Métrica de negócio: custo evitado de churn
- Churn observado na base: 26,54%

## Riscos

- Falso negativo: cliente com risco real não é tratado
- Falso positivo: cliente recebe ação de retenção sem necessidade
- Hipótese inicial: falso negativo custa mais caro
- Hipótese de custo inicial: FP = R$ 20 e FN = R$ 200

## SLOs iniciais

- API com disponibilidade adequada para inferência online
- Validação de schema antes do treino
- Pipeline reprodutível com seeds fixas e tracking no MLflow
- Endpoint `/predict` deve retornar probabilidade, classe e threshold

## Direcionadores observados na EDA

- Contratos `Month-to-month` têm churn muito superior aos contratos de maior fidelização.
- Clientes com `InternetService = Fiber optic` concentram risco elevado de churn.
- `Electronic check` aparece como o método de pagamento com maior taxa de saída.
