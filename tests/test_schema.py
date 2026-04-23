from __future__ import annotations

from tech_challenge.data.schema import build_input_schema


def test_telco_schema_accepts_expected_dataframe(telco_dataframe):
    schema = build_input_schema()
    validated = schema.validate(telco_dataframe)
    assert len(validated) == len(telco_dataframe)
