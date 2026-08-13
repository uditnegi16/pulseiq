"""Sentiment classification endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from pulseiq.api.deps import cache_ttl, get_cache_dep, get_sentiment_model
from pulseiq.api.models import SentimentPrediction, SentimentRequest, SentimentResponse
from pulseiq.llm.cache import make_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

MODEL_NAME = "distilbert-base-uncased + LoRA (PulseIQ fine-tune)"


@router.post("", response_model=SentimentResponse)
def classify(request: SentimentRequest, cache=Depends(get_cache_dep)) -> SentimentResponse:
    """Classify review sentiment.

    Confidence near 0.5 is meaningful, not a defect: 3-star reviews were
    excluded from training (see decision-log D-013), so genuinely mixed text has
    no class to belong to and the model correctly signals uncertainty.
    """
    key = make_key("sentiment", *request.texts)
    if (cached := cache.get(key)) is not None:
        return SentimentResponse(**{**cached, "cached": True})

    try:
        get_sentiment_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Sentiment model unavailable: {exc}",
        ) from exc

    from pulseiq.training.sentiment.predict import predict

    try:
        results = predict(request.texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("sentiment inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    response = SentimentResponse(
        predictions=[
            SentimentPrediction(text=text, **result)
            for text, result in zip(request.texts, results, strict=True)
        ],
        model_name=MODEL_NAME,
    )

    cache.set(key, response.model_dump(mode="json"), ttl=cache_ttl())
    return response
