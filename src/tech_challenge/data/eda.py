from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from tech_challenge.config import Settings
from tech_challenge.data.io import load_raw_dataset, prepare_dataset


def _save_plot(settings: Settings, fig: plt.Figure, output_name: str) -> str:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.reports_dir / output_name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _top_category_churn(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        dataframe.groupby(column, observed=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean())
        .sort_values(ascending=False)
        .rename("churn_rate")
        .reset_index()
    )
    grouped["churn_rate_pct"] = (grouped["churn_rate"] * 100).round(2)
    return grouped


def generate_eda_artifacts(settings: Settings) -> dict[str, object]:
    raw_dataframe = load_raw_dataset(settings)
    dataframe = prepare_dataset(raw_dataframe)

    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    missing_total_charges = int(dataframe["TotalCharges"].isna().sum())
    churn_distribution = dataframe["Churn"].value_counts(normalize=True).mul(100).round(2).to_dict()

    summary = {
        "raw_shape": {"rows": int(raw_dataframe.shape[0]), "columns": int(raw_dataframe.shape[1])},
        "modeling_shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
        "missing_total_charges": missing_total_charges,
        "raw_duplicate_rows": int(raw_dataframe.duplicated().sum()),
        "modeling_duplicate_rows": int(dataframe.duplicated().sum()),
        "churn_distribution_pct": churn_distribution,
        "numeric_summary": dataframe[numeric_columns].describe().round(2).to_dict(),
        "contract_churn": _top_category_churn(dataframe, "Contract").to_dict(orient="records"),
        "internet_service_churn": _top_category_churn(dataframe, "InternetService").to_dict(
            orient="records"
        ),
        "payment_method_churn": _top_category_churn(dataframe, "PaymentMethod").to_dict(
            orient="records"
        ),
    }

    histogram_fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, column in zip(axes, numeric_columns, strict=True):
        dataframe[column].dropna().plot(kind="hist", bins=30, ax=axis, title=column)
        axis.set_xlabel(column)
    histogram_path = _save_plot(settings, histogram_fig, "eda_numeric_distributions.png")

    contract_rates = _top_category_churn(dataframe, "Contract")
    contract_fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(contract_rates["Contract"], contract_rates["churn_rate_pct"])
    axis.set_ylabel("Churn rate (%)")
    axis.set_title("Churn rate by contract")
    contract_path = _save_plot(settings, contract_fig, "eda_contract_churn.png")

    payment_rates = _top_category_churn(dataframe, "PaymentMethod")
    payment_fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(payment_rates["PaymentMethod"], payment_rates["churn_rate_pct"])
    axis.set_ylabel("Churn rate (%)")
    axis.set_title("Churn rate by payment method")
    axis.tick_params(axis="x", rotation=20)
    payment_path = _save_plot(settings, payment_fig, "eda_payment_churn.png")

    internet_rates = _top_category_churn(dataframe, "InternetService")
    internet_fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(internet_rates["InternetService"], internet_rates["churn_rate_pct"])
    axis.set_ylabel("Churn rate (%)")
    axis.set_title("Churn rate by internet service")
    internet_path = _save_plot(settings, internet_fig, "eda_internet_churn.png")

    summary["artifacts"] = {
        "numeric_distributions": histogram_path,
        "contract_churn": contract_path,
        "payment_churn": payment_path,
        "internet_churn": internet_path,
    }

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.reports_dir / "eda_summary.json"
    markdown_path = settings.reports_dir / "eda_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = f"""# EDA Summary

## Dataset overview

- Raw rows: {summary["raw_shape"]["rows"]}
- Raw columns: {summary["raw_shape"]["columns"]}
- Modeling columns after dropping `customerID`: {summary["modeling_shape"]["columns"]}
- Duplicate rows in raw dataset: {summary["raw_duplicate_rows"]}
- Repeated feature profiles after dropping `customerID`: {summary["modeling_duplicate_rows"]}
- Missing `TotalCharges`: {summary["missing_total_charges"]}
- Churn rate: {summary["churn_distribution_pct"].get("Yes", 0)}%

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

- Numeric distributions: `{histogram_path}`
- Churn by contract: `{contract_path}`
- Churn by payment method: `{payment_path}`
- Churn by internet service: `{internet_path}`
"""
    markdown_path.write_text(markdown, encoding="utf-8")

    return summary


def main() -> None:
    summary = generate_eda_artifacts(Settings())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
