"""Health endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from pulseiq.api.models import ComponentHealth, HealthResponse, HealthStatus

router = APIRouter(tags=["health"])

VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service and dependency status.

    Returns `degraded` rather than `unavailable` when optional components are
    down. The forecast endpoint works without Mongo, Redis or an LLM key, so
    reporting a hard failure would pull the service out of a load balancer for
    something it can operate without.
    """
    components: list[ComponentHealth] = []

    # Relational store -- required.
    try:
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import count_rows

        engine = get_engine()
        init_db(engine)
        with session_scope(engine) as session:
            counts = count_rows(session)
        components.append(
            ComponentHealth(
                name="database",
                status=HealthStatus.OK,
                detail=", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            )
        )
        database_ok = True
    except Exception as exc:  # noqa: BLE001 - health checks report, never raise
        components.append(
            ComponentHealth(name="database", status=HealthStatus.UNAVAILABLE, detail=str(exc)[:120])
        )
        database_ok = False

    # Cache -- optional, always present in some form.
    from pulseiq.llm.cache import get_cache

    cache = get_cache()
    components.append(
        ComponentHealth(
            name="cache",
            status=HealthStatus.OK,
            detail=f"backend={cache.backend}"
            + (" (development fallback)" if cache.backend == "memory" else ""),
        )
    )

    # Sentiment model -- optional.
    try:
        from pulseiq.api.deps import get_sentiment_model

        get_sentiment_model()
        components.append(ComponentHealth(name="sentiment_model", status=HealthStatus.OK))
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentHealth(
                name="sentiment_model", status=HealthStatus.UNAVAILABLE, detail=str(exc)[:120]
            )
        )

    # LLM providers -- optional.
    from pulseiq.llm.router import available_providers

    providers = available_providers()
    components.append(
        ComponentHealth(
            name="llm",
            status=HealthStatus.OK if providers else HealthStatus.UNAVAILABLE,
            detail=", ".join(p.name for p in providers) or "no provider configured",
        )
    )

    if not database_ok:
        status = HealthStatus.UNAVAILABLE
    elif any(c.status is HealthStatus.UNAVAILABLE for c in components):
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.OK

    return HealthResponse(status=status, version=VERSION, components=components)


@router.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "pulseiq", "version": VERSION, "docs": "/docs"}
