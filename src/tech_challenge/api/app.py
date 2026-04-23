from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Request

from tech_challenge.api.schemas import HealthResponse, PredictRequest, PredictResponse
from tech_challenge.logging_config import configure_logging

if TYPE_CHECKING:
    from tech_challenge.api.service import PredictionService


configure_logging()
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Tech Challenge Churn API", version="0.1.0")


def prediction_service_dependency() -> PredictionService:
    from tech_challenge.api.service import get_prediction_service

    return get_prediction_service()


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    request_id = str(uuid4())
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    LOGGER.info(
        "request.completed",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "latency_ms": latency_ms,
            "status_code": response.status_code,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from tech_challenge.api.service import model_ready, model_version
    from tech_challenge.config import Settings

    settings = Settings()
    ready = model_ready(settings)
    return HealthResponse(
        status="ok",
        model_ready=ready,
        model_version=model_version(settings) if ready else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    service: Annotated[Any, Depends(prediction_service_dependency)],
) -> PredictResponse:
    prediction = service.predict(payload)
    return PredictResponse(**prediction)


def main() -> None:
    uvicorn.run("tech_challenge.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
