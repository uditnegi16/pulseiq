"""Shared pytest fixtures and configuration.

Integration tests are registered here rather than in each file so the marker is
defined once and `pytest -m "not integration"` works everywhere.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: exercises several real components together (slower)",
    )
    config.addinivalue_line("markers", "slow: long-running (model fits, downloads)")


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point every external resource at a temporary location.

    Without this, a test run writes to the developer's real pulseiq.db and
    inherits whatever credentials happen to be in .env -- which is how a test
    starts passing or failing based on the machine it runs on (see E-013).
    """
    from config.settings import settings
    from pulseiq.api import deps
    from pulseiq.llm.cache import reset_cache

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(settings, "redis_url", None)
    monkeypatch.setattr(settings, "mongodb_uri", None)
    monkeypatch.setattr(settings, "groq_api_key", None)
    monkeypatch.setattr(settings, "nvidia_nim_api_key", None)

    reset_cache()
    deps.reset_model_cache()
    yield tmp_path
    reset_cache()
    deps.reset_model_cache()
