"""Explicit configuration: importing this module never connects to a database."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> URL:
    """Read DATABASE_URL and select the supported PostgreSQL driver.

    Returns:
        Parsed SQLAlchemy URL using postgresql+psycopg. No connection is opened.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
        ValueError: If the URL specifies a database/driver this backend does not support.
        sqlalchemy.exc.ArgumentError: If the configured URL cannot be parsed.
    """
    configured_url = os.environ.get("DATABASE_URL")
    if not configured_url:
        raise RuntimeError("Set DATABASE_URL to a PostgreSQL connection URL.")
    parsed_url = make_url(configured_url)
    if parsed_url.drivername not in {"postgresql", "postgresql+psycopg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL with the psycopg driver.")
    return parsed_url.set(drivername="postgresql+psycopg")


def build_engine() -> Engine:
    """Configure a PostgreSQL connection pool without immediately connecting.

    Returns:
        SQLAlchemy Engine using DATABASE_URL, pool health checks and UTC sessions.
        The owner must dispose the engine at shutdown.

    Raises:
        RuntimeError: If DATABASE_URL is missing.
        ValueError: If the configured database/driver is unsupported.
        sqlalchemy.exc.ArgumentError: If the URL is malformed.
    """
    return create_engine(
        get_database_url(), pool_pre_ping=True, connect_args={"options": "-c timezone=UTC"}
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create ORM sessions backed by the supplied engine.

    Args:
        engine: Configured PostgreSQL engine; its ownership remains with the caller.

    Returns:
        Session factory with expire_on_commit=False. Callers must open/close each
        session and explicitly choose commit or rollback; this function writes nothing.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)
