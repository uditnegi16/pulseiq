"""LLM-backed strategy recommendations."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from pulseiq.api.deps import cache_ttl, get_cache_dep, get_db
from pulseiq.api.models import RecommendRequest, RecommendResponse
from pulseiq.llm.cache import make_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
def recommend(
    request: RecommendRequest,
    session=Depends(get_db),
    cache=Depends(get_cache_dep),
) -> RecommendResponse:
    """Generate a pricing recommendation grounded in stored price history.

    Returns 503 rather than canned text when no provider is available. A
    recommendation endpoint that silently returns filler when the LLM is down is
    worse than one that fails: the caller cannot distinguish analysis from
    placeholder.
    """
    from pulseiq.llm.prompts import SYSTEM_PROMPT, build_recommendation_prompt
    from pulseiq.llm.router import NoProviderAvailable, generate
    from pulseiq.storage.repository import load_price_history

    key = make_key("recommend", request.product_name, request.context or "")
    if (cached := cache.get(key)) is not None:
        return RecommendResponse(**{**cached, "cached": True})

    frame = load_price_history(session, product_name=request.product_name)
    if frame.empty:
        raise HTTPException(
            status_code=404, detail=f"No price history for '{request.product_name}'."
        )

    ordered = frame.sort_values("observed_on")
    prices = ordered["selling_price"].to_numpy(dtype=float)
    change_pct = (
        float((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] else None
    )

    prompt = build_recommendation_prompt(
        product_name=request.product_name,
        n_observations=len(prices),
        last_price=float(prices[-1]),
        price_change_pct=change_pct,
        context=request.context,
    )

    try:
        completion = generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = RecommendResponse(
        product_name=request.product_name,
        recommendation=completion.text,
        provider=completion.provider,
    )

    cache.set(key, response.model_dump(mode="json"), ttl=cache_ttl())
    return response
