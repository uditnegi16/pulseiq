"""Relational storage for cleaned, queryable data.

SQLite locally, PostgreSQL in deployment -- same code, different DATABASE_URL.

The tables mirror the Pydantic models in schemas.py. The database-level UNIQUE
constraints deliberately duplicate the in-memory dedupe logic in validation.py:
in-memory dedupe only sees the current batch, so re-running yesterday's scrape
would insert duplicates without the constraint. Two layers, two different
failure modes covered.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from datetime import date as Date

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy import (
    Date as SQLDate,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class PriceSnapshotRow(Base):
    """One product's price on one day."""

    __tablename__ = "price_snapshots"
    __table_args__ = (UniqueConstraint("product_key", "observed_on", name="uq_price_product_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    product_name: Mapped[str] = mapped_column(String(500), nullable=False)
    # Lowercased name; the actual dedupe key, kept as a column so the UNIQUE
    # constraint is case-insensitive on every backend (SQLite's default
    # collation is not).
    product_key: Mapped[str] = mapped_column(String(500), nullable=False, index=True)

    product_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")

    selling_price: Mapped[float] = mapped_column(Float, nullable=False)
    original_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    discount_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)

    observed_on: Mapped[Date] = mapped_column(SQLDate, nullable=False, index=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )


class ReviewRow(Base):
    """One customer review."""

    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("product_key", "review_hash", name="uq_review_product_text"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    product_name: Mapped[str] = mapped_column(String(500), nullable=False)
    product_key: Mapped[str] = mapped_column(String(500), nullable=False, index=True)

    review_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Hash rather than the text itself: review bodies exceed the index length
    # limit on MySQL/Postgres, and hashing sidesteps that entirely.
    review_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    verified_purchase: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="unknown")

    observed_on: Mapped[Date | None] = mapped_column(SQLDate, nullable=True, index=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )


def get_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create an engine. Defaults to DATABASE_URL from settings."""
    url = url or settings.database_url
    kwargs: dict = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        # Allows use across threads (Streamlit and FastAPI both need this).
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> None:
    """Create tables if absent. Safe to call repeatedly."""
    Base.metadata.create_all(engine)
    logger.info("schema ensured on %s", engine.url.render_as_string(hide_password=True))


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on any exception."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
