"""API request and response schemas.

Separate from `storage/schemas.py` on purpose. Those are the *internal* record
shapes; these are the *public contract*. Coupling them would mean a database
column rename becomes a breaking API change, and internal fields
(`ingested_at`, `text_hash`) would leak into responses that have no use for them.

Every response carries enough context to be interpreted without a second call:
predictions include confidence, forecasts include the baseline they beat (or
did not), and anything model-backed names the model that produced it.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ComponentHealth(BaseModel):
    """One dependency's status."""

    name: str
    status: HealthStatus
    detail: str = ""


class HealthResponse(BaseModel):
    """Overall service health.

    `degraded` rather than `unavailable` when optional components (Mongo, Redis,
    the LLM) are down: the core forecast and sentiment endpoints still work, and
    reporting a hard failure would take the service out of a load balancer for
    something it can operate without.
    """

    status: HealthStatus
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)


# --- sentiment --------------------------------------------------------------


class SentimentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    texts: list[str] = Field(min_length=1, max_length=100)


class SentimentPrediction(BaseModel):
    """One classification.

    Confidence is returned because a downstream consumer needs to know when the
    model is unsure: 0.51 positive and 0.99 positive are the same label and very
    different facts. Reviews resembling the excluded 3-star band land near 0.5,
    and callers should be able to see that.
    """

    text: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    positive: float = Field(ge=0.0, le=1.0)


class SentimentResponse(BaseModel):
    predictions: list[SentimentPrediction]
    model_name: str
    cached: bool = False


# --- forecasting ------------------------------------------------------------


class ForecastRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    product_name: str = Field(min_length=1, max_length=500)
    horizon: int = Field(default=3, ge=1, le=12, description="months ahead")
    model: str = Field(default="naive_last", description="see /forecast/models")


class ForecastPoint(BaseModel):
    period: int = Field(ge=1, description="steps ahead, 1 = next month")
    predicted_price: float


class ForecastResponse(BaseModel):
    """A forecast, with the honesty carried in the payload.

    `baseline_note` exists because this project's measured finding is that no
    model beats the naive forecast on this data (median h=1 MAE 0.0000). An API
    returning an ARIMA prediction without that context would imply a
    sophistication the evaluation does not support.
    """

    product_name: str
    model: str
    n_observations: int
    last_observed_price: float
    last_observed_date: date
    forecast: list[ForecastPoint]
    baseline_note: str
    cached: bool = False


class ProductSummary(BaseModel):
    product_name: str
    n_observations: int
    first_observed: date
    last_observed: date
    last_price: float


# --- recommendations --------------------------------------------------------


class RecommendRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    product_name: str = Field(min_length=1, max_length=500)
    context: str | None = Field(default=None, max_length=2000)


class RecommendResponse(BaseModel):
    product_name: str
    recommendation: str
    provider: str
    cached: bool = False


class ErrorResponse(BaseModel):
    detail: str
    hint: str | None = None
