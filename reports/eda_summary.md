# EDA Summary

## Dataset overview

- Raw rows: 7043
- Raw columns: 21
- Modeling columns after dropping `customerID`: 20
- Duplicate rows in raw dataset: 0
- Repeated feature profiles after dropping `customerID`: 22
- Missing `TotalCharges`: 11
- Churn rate: 26.54%

## Business insights

- Month-to-month contracts have the highest churn rate,
  reinforcing the retention risk of low-commitment plans.
- Fiber optic customers churn more than DSL customers in this dataset,
  suggesting service experience or price-pressure effects.
- Electronic check stands out as the payment method with the highest churn rate,
  which is a strong signal for proactive campaigns.

## Data quality interpretation

- The raw dataset has no exact duplicate rows.
- The 22 repeated rows appear only after removing `customerID`, so they likely represent
  different customers with the same profile rather than true duplicate records.

## Generated artifacts

- Numeric distributions: `C:\Tech Challenge\reports\eda_numeric_distributions.png`
- Churn by contract: `C:\Tech Challenge\reports\eda_contract_churn.png`
- Churn by payment method: `C:\Tech Challenge\reports\eda_payment_churn.png`
- Churn by internet service: `C:\Tech Challenge\reports\eda_internet_churn.png`
