"""HTTP client for the PulseIQ API.

The dashboard talks to the API rather than importing models directly. That
separation means the UI cannot accidentally depend on training internals, the
model loads once in the API process rather than once per Streamlit session, and
the same interface serves a UI, a script or a cron job unchanged.

Every method returns (data, error) instead of raising. A Streamlit page that
raises shows a red traceback to the user; one that gets an error string can show
a sentence explaining what to do.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class APIClient:
    def __init__(self, base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> tuple[Any | None, str | None]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
        except requests.ConnectionError:
            return None, (
                f"Cannot reach the API at {self.base_url}. "
                f"Start it with: uvicorn pulseiq.api.main:app --reload"
            )
        except requests.Timeout:
            return None, f"The API did not respond within {self.timeout:.0f}s."
        except requests.RequestException as exc:
            return None, f"Request failed: {type(exc).__name__}"

        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text[:200])
            except ValueError:
                detail = response.text[:200]
            return None, f"{response.status_code}: {detail}"

        try:
            return response.json(), None
        except ValueError:
            return None, "The API returned a response that was not valid JSON."

    def health(self):
        return self._request("GET", "/health")

    def list_products(self, limit: int = 100, min_observations: int = 8):
        return self._request(
            "GET",
            "/forecast/products",
            params={"limit": limit, "min_observations": min_observations},
        )

    def list_models(self):
        return self._request("GET", "/forecast/models")

    def forecast(self, product_name: str, horizon: int = 3, model: str = "naive_last"):
        return self._request(
            "POST",
            "/forecast",
            json={"product_name": product_name, "horizon": horizon, "model": model},
        )

    def price_history(self, product_name: str, limit: int = 12):
        """Recent observed prices, oldest first.

        Returns (prices, error) like every other method: a chart missing its
        history should degrade to showing only the forecast, not raise.
        """
        data, error = self._request(
            "GET",
            "/forecast/history",
            params={"product_name": product_name, "limit": limit},
        )
        if error or data is None:
            return None, error or "No response from API"
        return [point["price"] for point in data["history"]], None

    def sentiment(self, texts: list[str]):
        return self._request("POST", "/sentiment", json={"texts": texts})

    def recommend(self, product_name: str, context: str | None = None):
        return self._request(
            "POST", "/recommend", json={"product_name": product_name, "context": context}
        )
