"""Integration tests for the API against a real database and a real app instance.

The unit tests in tests/unit/test_api.py check each endpoint's contract. These
check that the endpoints work *together* on data that actually went through
ingestion -- the joins, not the pieces.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture
def client(isolated_settings):
    from pulseiq.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def seed(product: str = "Widget", n: int = 30, trend: float = 0.0, start: float = 20.0) -> None:
    from pulseiq.storage.relational import get_engine, init_db, session_scope
    from pulseiq.storage.repository import save_price_snapshots
    from pulseiq.storage.schemas import PriceSnapshot

    price = start
    records = []
    for i in range(n):
        price *= 1 + trend
        records.append(
            PriceSnapshot(
                product_name=product,
                selling_price=round(price, 2),
                observed_on=date.today() - timedelta(days=(n - i) * 20),
                source="integration_test",
            )
        )
    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as session:
        save_price_snapshots(session, records)


class TestDiscoveryToForecastFlow:
    """The journey a real client takes: what exists, what can I ask for, ask."""

    def test_products_then_models_then_forecast(self, client):
        seed("Widget", n=30)

        products = client.get("/forecast/products").json()
        assert products
        name = products[0]["product_name"]

        models = client.get("/forecast/models").json()
        assert "naive_last" in models

        for model in models:
            response = client.post(
                "/forecast", json={"product_name": name, "horizon": 2, "model": model}
            )
            assert response.status_code == 200, f"{model} failed: {response.text[:200]}"
            assert len(response.json()["forecast"]) == 2

    def test_product_names_round_trip_exactly(self, client):
        """Open Prices identifiers look like `3001234567890@101`. If the API
        mangles the name it returns, a client cannot feed it back in."""
        seed("3001234567890@101", n=30)

        name = client.get("/forecast/products").json()[0]["product_name"]
        assert name == "3001234567890@101"

        response = client.post("/forecast", json={"product_name": name})
        assert response.status_code == 200
        assert response.json()["product_name"] == name

    def test_min_observations_filter_is_honoured(self, client):
        seed("Long", n=30)
        seed("Short", n=4)

        permissive = client.get("/forecast/products", params={"min_observations": 2}).json()
        strict = client.get("/forecast/products", params={"min_observations": 20}).json()

        assert {p["product_name"] for p in strict} == {"Long"}
        assert len(permissive) >= len(strict)


class TestForecastCorrectness:
    def test_naive_repeats_the_last_observed_price(self, client):
        """The strongest measured model, and the easiest to verify: its output
        must equal the last observed price exactly."""
        seed("Flat", n=30, trend=0.0)

        payload = client.post(
            "/forecast", json={"product_name": "Flat", "horizon": 3, "model": "naive_last"}
        ).json()

        last = payload["last_observed_price"]
        assert all(
            p["predicted_price"] == pytest.approx(last, abs=0.01) for p in payload["forecast"]
        )

    def test_drift_extrapolates_an_upward_trend(self, client):
        seed("Rising", n=30, trend=0.02)

        payload = client.post(
            "/forecast", json={"product_name": "Rising", "horizon": 3, "model": "drift"}
        ).json()

        prices = [p["predicted_price"] for p in payload["forecast"]]
        assert prices == sorted(prices), "drift did not follow the trend"
        assert prices[-1] > payload["last_observed_price"]

    def test_longer_horizons_return_more_points(self, client):
        seed("Widget", n=30)
        for horizon in (1, 6, 12):
            payload = client.post(
                "/forecast", json={"product_name": "Widget", "horizon": horizon}
            ).json()
            assert len(payload["forecast"]) == horizon


class TestCacheBehaviour:
    def test_cache_is_scoped_per_model(self, client):
        """A cache key ignoring the model would serve one model's answer for
        another's request -- silently wrong, and invisible without this test."""
        seed("Rising", n=30, trend=0.02)

        naive = client.post(
            "/forecast", json={"product_name": "Rising", "model": "naive_last", "horizon": 3}
        ).json()
        drift = client.post(
            "/forecast", json={"product_name": "Rising", "model": "drift", "horizon": 3}
        ).json()

        assert drift["cached"] is False
        assert drift["model"] == "drift"
        assert naive["forecast"] != drift["forecast"]

    def test_cache_is_scoped_per_horizon(self, client):
        seed("Widget", n=30)

        client.post("/forecast", json={"product_name": "Widget", "horizon": 3})
        longer = client.post("/forecast", json={"product_name": "Widget", "horizon": 6}).json()

        assert longer["cached"] is False
        assert len(longer["forecast"]) == 6


class TestErrorPaths:
    """A client should be able to act on every error without reading the source."""

    def test_unknown_product_points_at_the_discovery_endpoint(self, client):
        response = client.post("/forecast", json={"product_name": "does not exist"})
        assert response.status_code == 404
        assert "/forecast/products" in response.json()["detail"]

    def test_unknown_model_lists_the_valid_options(self, client):
        seed("Widget", n=30)
        response = client.post(
            "/forecast", json={"product_name": "Widget", "model": "wishful_thinking"}
        )
        assert response.status_code == 400
        assert "naive_last" in response.json()["detail"]

    def test_insufficient_history_explains_the_requirement(self, client):
        seed("Sparse", n=3)
        response = client.post("/forecast", json={"product_name": "Sparse"})
        assert response.status_code in {404, 422}
        if response.status_code == 422:
            assert "too few" in response.json()["detail"].lower()

    def test_missing_llm_key_names_the_variable_to_set(self, client):
        seed("Widget", n=30)
        response = client.post("/recommend", json={"product_name": "Widget"})
        assert response.status_code == 503
        assert "GROQ_API_KEY" in response.json()["detail"]


class TestOpenAPIContract:
    def test_schema_documents_every_endpoint(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/health", "/forecast", "/sentiment", "/recommend"} <= set(paths)

    def test_forecast_response_schema_includes_the_baseline_caveat(self, client):
        """The measured finding is part of the published contract, not an
        implementation detail that could be dropped silently (D-019)."""
        schema = client.get("/openapi.json").json()
        properties = schema["components"]["schemas"]["ForecastResponse"]["properties"]
        assert "baseline_note" in properties
