from __future__ import annotations

import pandera.pandas as pa
from pandera import Check


def build_input_schema(target_column: str = "Churn") -> pa.DataFrameSchema:
    base_columns = {
        "gender": pa.Column(str),
        "SeniorCitizen": pa.Column(int, checks=Check.isin([0, 1])),
        "Partner": pa.Column(str),
        "Dependents": pa.Column(str),
        "tenure": pa.Column(int, checks=Check.ge(0)),
        "PhoneService": pa.Column(str),
        "MultipleLines": pa.Column(str),
        "InternetService": pa.Column(str),
        "OnlineSecurity": pa.Column(str),
        "OnlineBackup": pa.Column(str),
        "DeviceProtection": pa.Column(str),
        "TechSupport": pa.Column(str),
        "StreamingTV": pa.Column(str),
        "StreamingMovies": pa.Column(str),
        "Contract": pa.Column(str),
        "PaperlessBilling": pa.Column(str),
        "PaymentMethod": pa.Column(str),
        "MonthlyCharges": pa.Column(float, checks=Check.ge(0)),
        "TotalCharges": pa.Column(float, nullable=True, checks=Check.ge(0)),
    }
    if target_column:
        base_columns[target_column] = pa.Column(str, checks=Check.isin(["Yes", "No"]))
    return pa.DataFrameSchema(base_columns, strict=False)
