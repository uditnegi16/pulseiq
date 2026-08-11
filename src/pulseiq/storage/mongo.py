"""MongoDB Atlas client for raw scraped documents.

Why two stores: raw HTML-derived documents are schemaless, nested, and change
shape whenever a site changes its markup. Forcing them into columns loses
information you may need later. Cleaned, typed rows go to the relational store
for querying; the raw document is kept here as the audit trail.

That split is the honest answer to "why MongoDB?" -- a question worth being able
to answer with something better than "the JD mentioned it".

Everything degrades gracefully when MONGODB_URI is unset, so the pipeline runs
locally with SQLite alone.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

_client: Any = None


class MongoNotConfigured(RuntimeError):
    """Raised when a Mongo operation is attempted without MONGODB_URI set."""


def is_configured() -> bool:
    return settings.mongodb_uri is not None


def get_client(*, uri: str | None = None, timeout_ms: int = 5000) -> Any:
    """Return a cached MongoClient.

    A short serverSelectionTimeoutMS matters: pymongo's default is 30s, so a
    wrong URI hangs the whole ingestion run before failing.
    """
    global _client
    if uri is None and not is_configured():
        raise MongoNotConfigured(
            "MONGODB_URI is not set. Add it to .env (see .env.example) or run with --no-mongo."
        )

    if uri is not None:
        from pymongo import MongoClient

        return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)

    if _client is None:
        from pymongo import MongoClient

        _client = MongoClient(settings.require("mongodb_uri"), serverSelectionTimeoutMS=timeout_ms)
    return _client


def get_raw_collection(client: Any = None) -> Any:
    client = client or get_client()
    return client[settings.mongodb_db][settings.mongodb_raw_collection]


def build_raw_document(
    payload: dict[str, Any], *, source: str, run_id: str | None = None
) -> dict[str, Any]:
    """Wrap a scraped payload with ingestion metadata. Pure -- no I/O.

    The envelope (not the payload) is what makes the collection queryable:
    "show me everything from the run that broke" needs run_id, and drift
    analysis needs ingested_at.
    """
    return {
        "source": source,
        "run_id": run_id,
        "ingested_at": datetime.now(UTC),
        "payload": payload,
    }


def insert_raw_documents(
    payloads: Iterable[dict[str, Any]],
    *,
    source: str,
    run_id: str | None = None,
    collection: Any = None,
) -> int:
    """Insert raw scrape payloads. Returns the number inserted.

    `collection` is injectable so this is testable with a fake, and callers can
    reuse one handle across a run.
    """
    docs = [build_raw_document(p, source=source, run_id=run_id) for p in payloads]
    if not docs:
        return 0

    target = collection if collection is not None else get_raw_collection()
    result = target.insert_many(docs)
    count = len(result.inserted_ids)
    logger.info("inserted %d raw documents into mongo (run_id=%s)", count, run_id)
    return count


def ping(client: Any = None) -> bool:
    """Check connectivity. Returns False rather than raising, so a startup
    health check can report status without crashing."""
    try:
        client = client or get_client()
        client.admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001 - health checks must not propagate
        logger.warning("mongo ping failed: %s", exc)
        return False


def reset_client() -> None:
    """Drop the cached client. Used by test fixtures."""
    global _client
    _client = None
