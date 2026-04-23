from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def telco_dataframe() -> pd.DataFrame:
    records = []
    for idx in range(60):
        records.append(
            {
                "gender": "Female" if idx % 2 == 0 else "Male",
                "SeniorCitizen": idx % 2,
                "Partner": "Yes" if idx % 3 else "No",
                "Dependents": "No" if idx % 4 else "Yes",
                "tenure": 6 + idx,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if idx % 2 else "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month" if idx % 2 else "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0 + idx,
                "TotalCharges": 300.0 + 25 * idx,
                "Churn": "Yes" if idx % 3 == 0 else "No",
            }
        )
    return pd.DataFrame.from_records(records)
