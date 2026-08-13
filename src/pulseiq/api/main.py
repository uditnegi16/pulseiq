"""FastAPI application factory.

The Streamlit dashboard consumes this API rather than importing models directly.
That separation is deliberate: it means the dashboard cannot accidentally depend
on training-time internals, the model loads once in one process rather than once
per Streamlit session, and the same interface serves a UI, a script, or a cron
job without change.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from pulseiq.api.routers import forecast, health, recommend, sentiment

logger = logging.getLogger(__name__)

DESCRIPTION = """
Competitor pricing forecasts and review sentiment.

**Measured findings this API reflects:**

* Forecasting -- no model beats a naive "price stays the same" baseline on this
  data at any tested horizon (median h=1 MAE 0.0000). Every forecast response
  carries that caveat rather than implying unearned precision.
* Sentiment -- LoRA fine-tuning improved accuracy from 88.8% to 94.1%, almost
  entirely through recall (+11.5 points, precision flat).

See `docs/metrics.md` for protocol and limitations.
"""


def create_app() -> FastAPI:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,  # stdout only: CloudWatch captures it, files do not exist on Lambda
    )

    app = FastAPI(
        title="PulseIQ API",
        description=DESCRIPTION,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Permissive CORS is acceptable here: the API serves public, non-personal
    # data and holds no session state. Tighten before it ever serves anything else.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(sentiment.router)
    app.include_router(recommend.router)

    return app


app = create_app()
