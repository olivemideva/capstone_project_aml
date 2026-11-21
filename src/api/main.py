"""FastAPI service for demographic reconstruction and simulated prioritisation."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.metrics import ApiMetrics
from src.api.schemas import (
    County,
    Indicators,
    PrioritiseRequest,
    PrioritiseResponse,
    ReconstructionRequest,
    ReconstructionResponse,
)
from src.api.service import SOURCE, Service
from src.data.load import COUNTY_CODES

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def create_app(
    model_dir: Path | None = None,
    indicator_path: Path | None = None,
    expected_counties: set[str] | None = COUNTY_CODES,
) -> FastAPI:
    selected = model_dir or Path(os.environ.get("AML_MODEL_DIR", ROOT / "models" / "selected"))
    indicators = indicator_path or Path(
        os.environ.get("AML_INDICATORS_PATH", ROOT / "models" / "observed_county_indicators.csv")
    )
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            service = Service.load(selected, indicators)
            if (
                expected_counties is not None
                and set(service.indicators.admin1_code) != expected_counties
            ):
                raise ValueError(
                    "Indicator artifact must contain exactly the expected county codes"
                )
            app.state.service = service
        except Exception:
            LOGGER.exception("Failed to load API artifacts; service cannot start")
            raise
        yield

    app = FastAPI(
        title="Kenya Demographic Decision Support API", version="0.1.0", lifespan=lifespan
    )
    origins = [origin.strip() for origin in os.environ.get("AML_CORS_ORIGINS", "").split(",")]
    origins = [origin for origin in origins if origin]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
        )
    app.middleware("http")(metrics.middleware)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_family": request.app.state.service.family}

    @app.get("/counties", response_model=list[County])
    def counties(request: Request):
        data = request.app.state.service.indicators
        return data[["admin1_code", "admin1_name"]].to_dict(orient="records")

    @app.get("/indicators", response_model=list[Indicators])
    def all_indicators(request: Request):
        data = request.app.state.service.indicators
        return [{**row, "source": SOURCE} for row in data.to_dict(orient="records")]

    @app.get("/indicators/{county}", response_model=Indicators)
    def indicators_for_county(county: str, request: Request):
        data = request.app.state.service.indicators
        matches = data[data["admin1_code"].str.casefold() == county.strip().casefold()]
        if matches.empty:
            raise HTTPException(status_code=404, detail="County code not found")
        return {**matches.iloc[0].to_dict(), "source": SOURCE}

    @app.post("/reconstruct", response_model=ReconstructionResponse)
    def reconstruct(payload: ReconstructionRequest, request: Request):
        service = request.app.state.service
        result, predicted = service.reconstruct(payload.population, payload.pattern, payload.rate)
        metrics.predictions.inc()
        return {
            "population": result,
            "predicted_fields": predicted,
            "model_family": service.family,
            "pattern": payload.pattern,
            "rate": payload.rate,
            "note": (
                "Experimental predictions; choose the scenario to match the missingness pattern."
            ),
        }

    @app.post("/prioritise", response_model=PrioritiseResponse)
    def prioritise(payload: PrioritiseRequest, request: Request):
        service = request.app.state.service
        if payload.budget > len(service.indicators):
            raise HTTPException(status_code=422, detail="Budget exceeds number of counties")
        if not service.optimization_lock.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Prioritisation is busy; retry later")
        try:
            solutions = service.prioritise(
                payload.budget, payload.weights, payload.seed, payload.generations
            )
        finally:
            service.optimization_lock.release()
        return {
            "budget": payload.budget,
            "weights": payload.weights,
            "seed": payload.seed,
            "generations": payload.generations,
            "source": SOURCE,
            "note": (
                "Simulated equal-cost selection, not an actual allocation; NSGA-III is approximate."
            ),
            "solutions": solutions,
        }

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return metrics.response()

    return app


app = create_app()
