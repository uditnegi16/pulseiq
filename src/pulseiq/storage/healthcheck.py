"""Storage health check: verify Mongo and SQL connectivity before a long run.

Fail fast, with a specific reason. The alternative is discovering a bad
connection string after a 30-second Parquet download and a validation pass.

Usage (project root, venv active):

    python -m pulseiq.storage.healthcheck
    python -m pulseiq.storage.healthcheck --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

from config.settings import settings

logger = logging.getLogger(__name__)

OK = "OK"
FAIL = "FAIL"
SKIP = "SKIP"


def _mask(uri: str | None) -> str:
    """Show enough of a URI to identify it, never enough to use it.

    Connection strings carry credentials, so this must not print one even into a
    local terminal that might end up in a screenshot or a pasted log.
    """
    if not uri:
        return "(not set)"
    if "@" in uri:
        scheme, _, rest = uri.partition("://")
        host = rest.split("@", 1)[1]
        return f"{scheme}://***:***@{host[:40]}"
    return uri[:60]


def check_relational(verbose: bool = False) -> tuple[str, str]:
    """Connect to the relational store and count rows."""
    try:
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import count_rows

        engine = get_engine()
        init_db(engine)
        with session_scope(engine) as session:
            counts = count_rows(session)
        detail = ", ".join(f"{table}={n}" for table, n in sorted(counts.items()))
        return OK, f"{_mask(settings.database_url)} | {detail}"
    except Exception as exc:  # noqa: BLE001 - a health check reports, never raises
        if verbose:
            logger.exception("relational check failed")
        return FAIL, f"{type(exc).__name__}: {str(exc)[:120]}"


def check_mongo(verbose: bool = False) -> tuple[str, str]:
    """Ping MongoDB and report the raw-document count."""
    from pulseiq.storage import mongo

    if not mongo.is_configured():
        return SKIP, "MONGODB_URI not set (pipeline runs on SQL alone)"

    try:
        client = mongo.get_client(timeout_ms=5000)
        if not mongo.ping(client):
            return FAIL, "ping failed -- check the URI, password, and IP allowlist"

        collection = mongo.get_raw_collection(client)
        count = collection.estimated_document_count()
        return OK, (
            f"{_mask(settings.require('mongodb_uri'))} | "
            f"db={settings.mongodb_db} collection={settings.mongodb_raw_collection} "
            f"docs~={count}"
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            logger.exception("mongo check failed")
        return FAIL, f"{type(exc).__name__}: {str(exc)[:120]}"


def check_mlflow(verbose: bool = False) -> tuple[str, str]:
    """Confirm the tracking store is reachable and not the removed file store."""
    uri = settings.mlflow_tracking_uri
    if uri.startswith("file:"):
        return FAIL, (f"{uri} -- MLflow 3.15 removed the filesystem store. Use sqlite:///mlflow.db")
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        experiments = mlflow.search_experiments()
        return OK, f"{uri} | {len(experiments)} experiment(s)"
    except Exception as exc:  # noqa: BLE001
        if verbose:
            logger.exception("mlflow check failed")
        return FAIL, f"{type(exc).__name__}: {str(exc)[:120]}"


def run(verbose: bool = False) -> int:
    """Run all checks. Returns 0 unless something actively failed.

    A SKIP is not a failure: Mongo is optional by design, and the pipeline is
    built to run without it.
    """
    checks = [
        ("relational", check_relational),
        ("mongodb", check_mongo),
        ("mlflow", check_mlflow),
    ]

    print(f"{'component':<14}{'status':<8}detail")
    print("-" * 92)

    failed = 0
    for name, check in checks:
        status, detail = check(verbose)
        if status == FAIL:
            failed += 1
        print(f"{name:<14}{status:<8}{detail}")

    print()
    if failed:
        print(f"{failed} check(s) failed. See .env.example for the expected variables.")
        return 1
    print("all configured components reachable.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.storage.healthcheck",
        description="Verify storage and tracking connectivity.",
    )
    parser.add_argument("--verbose", action="store_true", help="print full tracebacks")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    return run(args.verbose)


if __name__ == "__main__":
    sys.exit(main())
