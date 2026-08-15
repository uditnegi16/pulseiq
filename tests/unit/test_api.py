"""Tests for the serving layer: API endpoints, cache, and LLM routing.

No network, no live Redis, no LLM key. The router is tested with a fake session
object, and the API with a temporary SQLite database seeded per test.

A note on type checking: the fakes here (FakeSession, BrokenClient) deliberately
do not satisfy the production type signatures -- a stand-in that fully implemented
`requests.Session` would not be a stand-in. Pyright is configured to skip this
directory for that reason; the code under test in src/ is checked normally.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from pulseiq.llm.cache import InMemoryCache, RedisCache, get_cache, make_key, reset_cache
from pulseiq.llm.prompts import (
    SYSTEM_PROMPT,
    build_recommendation_prompt,
    build_sentiment_summary,
)
from pulseiq.llm.router import (
    Completion,
    NoProviderAvailable,
    Provider,
    TransientProviderError,
    available_providers,
    call_provider,
    generate,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """API client backed by a temporary SQLite database."""
    from config.settings import settings
    from pulseiq.api import deps

    # get_engine() builds a fresh engine per call, so pointing DATABASE_URL at a
    # temp file is enough to isolate each test -- no engine cache to clear.
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    reset_cache()
    deps.reset_model_cache()

    from pulseiq.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client

    reset_cache()
    deps.reset_model_cache()


def seed_prices(product="Widget", n=30, start_price=20.0, trend=0.0):
    """Insert a price series into whatever database is configured."""
    from pulseiq.storage.relational import get_engine, init_db, session_scope
    from pulseiq.storage.repository import save_price_snapshots
    from pulseiq.storage.schemas import PriceSnapshot

    price = start_price
    records = []
    for i in range(n):
        price *= 1 + trend
        records.append(
            PriceSnapshot(
                product_name=product,
                selling_price=round(price, 2),
                observed_on=date.today() - timedelta(days=(n - i) * 20),
                source="test",
            )
        )
    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as session:
        save_price_snapshots(session, records)


# --- cache ------------------------------------------------------------------


class TestMakeKey:
    def test_deterministic(self):
        assert make_key("ns", "a", 1) == make_key("ns", "a", 1)

    def test_varies_with_every_component(self):
        """A key that ignores a parameter returns another request's answer."""
        base = make_key("forecast", "Widget", "naive_last", 3)
        assert base != make_key("forecast", "Widget", "arima_auto", 3)
        assert base != make_key("forecast", "Widget", "naive_last", 6)
        assert base != make_key("forecast", "Other", "naive_last", 3)

    def test_long_unicode_input_produces_a_short_ascii_key(self):
        key = make_key("sentiment", "é" * 5000)
        assert len(key) < 100
        assert key.isascii()

    def test_user_content_is_not_embedded_in_the_key(self):
        assert "secret review text" not in make_key("sentiment", "secret review text")


class TestInMemoryCache:
    def test_round_trip(self):
        cache = InMemoryCache()
        cache.set("k", {"a": 1})
        assert cache.get("k") == {"a": 1}

    def test_missing_key_returns_none(self):
        assert InMemoryCache().get("nope") is None

    def test_ttl_expires(self):
        cache = InMemoryCache()
        cache.set("k", 1, ttl=1)
        assert cache.get("k") == 1
        time.sleep(1.1)
        assert cache.get("k") is None

    def test_is_bounded(self):
        """An unbounded process-local cache is a memory leak with a nice name."""
        cache = InMemoryCache(max_entries=5)
        for i in range(20):
            cache.set(f"k{i}", i)
        assert len(cache._store) <= 5

    def test_delete(self):
        cache = InMemoryCache()
        cache.set("k", 1)
        cache.delete("k")
        assert cache.get("k") is None

    def test_backend_name(self):
        assert InMemoryCache().backend == "memory"


class TestCacheSelection:
    def test_falls_back_to_memory_without_redis(self, monkeypatch):
        """Requiring Redis to start would end the project's zero-credential property."""
        from config.settings import settings

        monkeypatch.setattr(settings, "redis_url", None)
        reset_cache()
        assert get_cache().backend == "memory"
        reset_cache()

    def test_redis_failure_falls_back_rather_than_crashing(self, monkeypatch):
        """An unreachable Redis must not stop the API from starting."""
        from pydantic import SecretStr

        from config.settings import settings
        from pulseiq.llm import cache as cache_module

        # SecretStr, not a plain string: `settings.require` unwraps it, and
        # pydantic will not allow patching a method on the model itself.
        monkeypatch.setattr(settings, "redis_url", SecretStr("redis://nonexistent.invalid:6379"))

        def explode(*args, **kwargs):
            raise ConnectionError("no route to host")

        monkeypatch.setattr(cache_module, "RedisCache", explode)
        reset_cache()
        assert get_cache().backend == "memory"
        reset_cache()


class TestRedisCacheErrorHandling:
    """A cache is an optimisation. If it is down the request should still be
    served, just slower -- propagating a timeout as a 500 makes the cache a
    liability."""

    @staticmethod
    def _broken_cache(monkeypatch):
        cache = RedisCache.__new__(RedisCache)

        class BrokenClient:
            def get(self, *a, **k):
                raise ConnectionError("down")

            def set(self, *a, **k):
                raise ConnectionError("down")

            def setex(self, *a, **k):
                raise ConnectionError("down")

            def delete(self, *a, **k):
                raise ConnectionError("down")

            def ping(self):
                raise ConnectionError("down")

        cache._client = BrokenClient()
        return cache

    def test_get_failure_is_a_miss(self, monkeypatch):
        assert self._broken_cache(monkeypatch).get("k") is None

    def test_set_failure_does_not_raise(self, monkeypatch):
        self._broken_cache(monkeypatch).set("k", 1, ttl=10)

    def test_ping_failure_returns_false(self, monkeypatch):
        assert self._broken_cache(monkeypatch).ping() is False


# --- prompts ----------------------------------------------------------------


class TestPrompts:
    def test_system_prompt_forbids_price_prediction(self):
        """The project measured that no model beats naive. A recommendation
        confidently predicting prices would contradict its own findings."""
        assert "do not predict specific future prices" in SYSTEM_PROMPT.lower()

    def test_includes_supplied_evidence(self):
        prompt = build_recommendation_prompt(
            product_name="Widget", n_observations=42, last_price=19.99, price_change_pct=-8.3
        )
        assert "Widget" in prompt and "42" in prompt and "19.99" in prompt
        assert "down 8.3%" in prompt

    def test_omits_absent_fields_rather_than_filling_them(self):
        """A model shown 'sentiment: unknown' reasons about the unknown-ness."""
        prompt = build_recommendation_prompt(
            product_name="Widget", n_observations=10, last_price=5.0
        )
        assert "sentiment" not in prompt.lower()
        assert "trend" not in prompt.lower()

    def test_small_review_sample_is_flagged_not_summarised(self):
        assert "too few" in build_sentiment_summary(2, 0)

    def test_adequate_sample_gets_a_percentage(self):
        assert "80% positive" in build_sentiment_summary(80, 20)

    def test_zero_reviews(self):
        assert "no reviews" in build_sentiment_summary(0, 0)


# --- llm router -------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(url)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def completion_payload(text="advice here"):
    return {"choices": [{"message": {"content": text}}]}


PROVIDER_A = Provider(name="a", url="https://a.invalid", api_key="k", model="m")
PROVIDER_B = Provider(name="b", url="https://b.invalid", api_key="k", model="m")


class TestCallProvider:
    def test_successful_call(self):
        session = FakeSession([FakeResponse(200, completion_payload("do X"))])
        result = call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)
        assert result.text == "do X"
        assert result.provider == "a"

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_transient_statuses_are_retryable(self, status):
        session = FakeSession([FakeResponse(status)])
        with pytest.raises(TransientProviderError):
            call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)

    def test_client_error_is_not_retryable(self):
        """Retrying a 400 elsewhere just produces the same 400 more slowly."""
        session = FakeSession([FakeResponse(400)])
        with pytest.raises(RuntimeError) as excinfo:
            call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)
        assert not isinstance(excinfo.value, TransientProviderError)

    def test_timeout_is_retryable(self):
        session = FakeSession([requests.Timeout()])
        with pytest.raises(TransientProviderError, match="timed out"):
            call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)

    def test_malformed_response_is_retryable(self):
        session = FakeSession([FakeResponse(200, {"unexpected": "shape"})])
        with pytest.raises(TransientProviderError, match="unexpected response"):
            call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)

    def test_empty_completion_is_retryable(self):
        session = FakeSession([FakeResponse(200, completion_payload("   "))])
        with pytest.raises(TransientProviderError, match="empty"):
            call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)

    def test_error_message_never_contains_the_api_key(self):
        """Provider errors can echo the request, and the request carries the
        Authorization header."""
        session = FakeSession([FakeResponse(403, text="denied for key sk-secret")])
        with pytest.raises(RuntimeError) as excinfo:
            call_provider(PROVIDER_A, system_prompt="s", user_prompt="u", session=session)
        assert "sk-secret" not in str(excinfo.value)
        assert PROVIDER_A.api_key not in str(excinfo.value)


class TestGenerate:
    def test_falls_over_to_the_second_provider(self, monkeypatch):
        monkeypatch.setattr(
            "pulseiq.llm.router.call_provider",
            lambda provider, **kw: (
                Completion("ok", provider.name, "m")
                if provider.name == "b"
                else (_ for _ in ()).throw(TransientProviderError("429"))
            ),
        )
        result = generate(system_prompt="s", user_prompt="u", providers=[PROVIDER_A, PROVIDER_B])
        assert result.provider == "b"

    def test_no_providers_raises_actionable_error(self):
        with pytest.raises(NoProviderAvailable, match="GROQ_API_KEY"):
            generate(system_prompt="s", user_prompt="u", providers=[])

    def test_all_providers_failing_lists_each_failure(self, monkeypatch):
        def always_fail(provider, **kwargs):
            raise TransientProviderError(f"{provider.name}: down")

        monkeypatch.setattr("pulseiq.llm.router.call_provider", always_fail)
        with pytest.raises(NoProviderAvailable) as excinfo:
            generate(system_prompt="s", user_prompt="u", providers=[PROVIDER_A, PROVIDER_B])
        assert "a: down" in str(excinfo.value)
        assert "b: down" in str(excinfo.value)

    def test_non_retryable_error_stops_immediately(self, monkeypatch):
        attempts = []

        def fail_hard(provider, **kwargs):
            attempts.append(provider.name)
            raise RuntimeError(f"{provider.name}: HTTP 400")

        monkeypatch.setattr("pulseiq.llm.router.call_provider", fail_hard)
        with pytest.raises(NoProviderAvailable):
            generate(system_prompt="s", user_prompt="u", providers=[PROVIDER_A, PROVIDER_B])
        assert attempts == ["a"]  # did not waste a call on b

    def test_no_providers_configured_without_keys(self, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "groq_api_key", None)
        monkeypatch.setattr(settings, "nvidia_nim_api_key", None)
        assert available_providers() == []


# --- api --------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200_even_when_optional_components_are_down(self, client):
        """Degraded, not unavailable -- the core endpoints still work without
        Mongo, Redis, or an LLM key."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in {"ok", "degraded"}

    def test_lists_every_component(self, client):
        names = {c["name"] for c in client.get("/health").json()["components"]}
        assert names == {"database", "cache", "sentiment_model", "llm"}

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_openapi_schema_generates(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestForecastEndpoint:
    def test_lists_available_models(self, client):
        models = client.get("/forecast/models").json()
        assert "naive_last" in models
        assert "arima_auto" in models

    def test_empty_database_returns_empty_product_list(self, client):
        assert client.get("/forecast/products").json() == []

    def test_unknown_product_returns_404(self, client):
        response = client.post("/forecast", json={"product_name": "nope", "horizon": 3})
        assert response.status_code == 404
        assert "No price history" in response.json()["detail"]

    def test_invalid_horizon_is_rejected(self, client):
        assert (
            client.post("/forecast", json={"product_name": "x", "horizon": 99}).status_code == 422
        )
        assert client.post("/forecast", json={"product_name": "x", "horizon": 0}).status_code == 422

    def test_unknown_model_returns_400(self, client):
        seed_prices("Widget", n=30)
        response = client.post(
            "/forecast", json={"product_name": "Widget", "horizon": 3, "model": "magic"}
        )
        assert response.status_code == 400
        assert "Unknown model" in response.json()["detail"]

    def test_forecast_returns_the_requested_horizon(self, client):
        seed_prices("Widget", n=30)
        result = client.post("/forecast", json={"product_name": "Widget", "horizon": 5}).json()
        assert len(result["forecast"]) == 5
        assert [p["period"] for p in result["forecast"]] == [1, 2, 3, 4, 5]

    def test_response_carries_the_baseline_caveat(self, client):
        """An API returning an ARIMA number without the caveat would imply a
        sophistication the evaluation does not support."""
        seed_prices("Widget", n=30)
        result = client.post("/forecast", json={"product_name": "Widget"}).json()
        assert "no model beats" in result["baseline_note"].lower()

    def test_second_identical_request_is_served_from_cache(self, client):
        seed_prices("Widget", n=30)
        first = client.post("/forecast", json={"product_name": "Widget", "horizon": 3}).json()
        second = client.post("/forecast", json={"product_name": "Widget", "horizon": 3}).json()
        assert first["cached"] is False
        assert second["cached"] is True

    def test_different_models_are_cached_separately(self, client):
        seed_prices("Trend", n=30, trend=0.01)
        client.post("/forecast", json={"product_name": "Trend", "model": "naive_last"})
        drift = client.post("/forecast", json={"product_name": "Trend", "model": "drift"}).json()
        assert drift["cached"] is False
        assert drift["model"] == "drift"

    def test_models_differ_on_a_trending_series(self, client):
        seed_prices("Trend", n=30, trend=0.01)
        naive = client.post(
            "/forecast", json={"product_name": "Trend", "horizon": 3, "model": "naive_last"}
        ).json()
        drift = client.post(
            "/forecast", json={"product_name": "Trend", "horizon": 3, "model": "drift"}
        ).json()
        assert naive["forecast"][-1]["predicted_price"] != drift["forecast"][-1]["predicted_price"]

    def test_too_little_history_returns_422(self, client):
        seed_prices("Sparse", n=2)
        response = client.post("/forecast", json={"product_name": "Sparse"})
        assert response.status_code in {404, 422}


class TestSentimentEndpoint:
    def test_missing_adapter_returns_503_not_500(self, client, monkeypatch):
        """A missing model is a service-availability problem, not a bug.

        The absence is simulated rather than relying on the adapter genuinely
        being absent: this test previously passed only on machines where nobody
        had downloaded models/sentiment_lora, and failed once the adapter was
        installed. A test whose outcome depends on whether an optional file
        happens to exist is not testing anything.
        """
        from pulseiq.api import deps

        def unavailable():
            raise RuntimeError("No adapter at models/sentiment_lora (simulated)")

        monkeypatch.setattr(deps, "get_sentiment_model", unavailable)
        monkeypatch.setattr("pulseiq.api.routers.sentiment.get_sentiment_model", unavailable)

        response = client.post("/sentiment", json={"texts": ["great product"]})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

    def test_classifies_when_the_adapter_is_present(self, client):
        """Runs only where inference can actually work, so CI skips it rather
        than failing on an optional artifact.

        The guard checks that the model LOADS, not merely that the adapter
        directory exists. A file-existence check would fail confusingly on a
        machine that has the adapter but no torch.
        """
        # Not pytest.importorskip: that only catches ImportError, and a broken
        # torch install raises OSError on the missing shared library instead.
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"torch unavailable: {str(exc)[:60]}")

        if not Path("models/sentiment_lora/adapter_config.json").exists():
            pytest.skip("adapter not downloaded -- see notebooks/finetune_sentiment.ipynb")

        from pulseiq.api.deps import get_sentiment_model

        try:
            get_sentiment_model()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"sentiment model cannot load here: {str(exc)[:80]}")

        response = client.post(
            "/sentiment",
            json={"texts": ["Battery died in a week, waste of money.", "Superb sound quality."]},
        )
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert predictions[0]["label"] == "negative"
        assert predictions[1]["label"] == "positive"
        assert all(0.0 <= p["confidence"] <= 1.0 for p in predictions)

    def test_empty_text_list_is_rejected(self, client):
        assert client.post("/sentiment", json={"texts": []}).status_code == 422

    def test_batch_size_is_capped(self, client):
        assert client.post("/sentiment", json={"texts": ["x"] * 500}).status_code == 422


class TestRecommendEndpoint:
    def test_unknown_product_returns_404(self, client):
        assert client.post("/recommend", json={"product_name": "nope"}).status_code == 404

    def test_no_llm_configured_returns_503_not_filler(self, client, monkeypatch):
        """Canned text when the LLM is down is worse than a failure -- the caller
        cannot tell analysis from placeholder."""
        from config.settings import settings

        monkeypatch.setattr(settings, "groq_api_key", None)
        monkeypatch.setattr(settings, "nvidia_nim_api_key", None)
        seed_prices("Widget", n=30)
        response = client.post("/recommend", json={"product_name": "Widget"})
        assert response.status_code == 503
        assert "GROQ_API_KEY" in response.json()["detail"]


class TestAPIClient:
    def test_connection_error_returns_actionable_message(self):
        from app.api_client import APIClient

        client = APIClient("http://localhost:59999", timeout=1.0)
        data, error = client.health()
        assert data is None
        assert error is not None
        assert "uvicorn" in error


class TestDashboardSourceHygiene:
    """Static checks on the Streamlit app.

    Streamlit code is hard to test by execution -- it renders rather than
    returns. These assert the two structural mistakes that actually happened,
    both of which were invisible until the page was opened in a browser.
    """

    @staticmethod
    def _dashboard_source() -> str:
        from pathlib import Path

        return (Path(__file__).parents[2] / "app" / "streamlit_app.py").read_text(encoding="utf-8")

    def test_no_conditional_expression_used_as_a_statement(self):
        """`st.success(...) if cond else st.warning(...)` renders the returned
        DeltaGenerator's repr into the page -- several hundred lines of API
        docs where a one-line status indicator should be."""
        import ast

        tree = ast.parse(self._dashboard_source())
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.IfExp)
        ]
        assert not offenders, (
            f"conditional expression used as a statement at line(s) {offenders}; "
            f"Streamlit will render the DeltaGenerator repr"
        )

    def test_dashboard_does_not_import_training_code(self):
        """The dashboard consumes the API over HTTP (D-018). An import from
        pulseiq.training would load the model per Streamlit session and couple
        the UI to training internals."""
        source = self._dashboard_source()
        assert "from pulseiq" not in source
        assert "import pulseiq" not in source

    def test_dashboard_parses(self):
        import ast

        ast.parse(self._dashboard_source())
