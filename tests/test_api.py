from __future__ import annotations

from fastapi.testclient import TestClient

from tech_challenge.api.app import app, prediction_service_dependency


class DummyService:
    def predict(self, request):
        del request
        return {
            "churn_probability": 0.73,
            "churn_prediction": 1,
            "threshold": 0.35,
            "model_version": "test",
        }


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_ready" in response.json()


def test_predict_endpoint():
    app.dependency_overrides[prediction_service_dependency] = lambda: DummyService()
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.9,
            "TotalCharges": 2112.4,
        },
    )
    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["churn_prediction"] == 1
