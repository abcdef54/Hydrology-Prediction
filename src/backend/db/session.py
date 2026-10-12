"""Explicit configuration: importing this module never connects to a database."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> URL:
    configured_url = os.environ.get("DATABASE_URL")
    if not configured_url:
        raise RuntimeError("Set DATABASE_URL to a PostgreSQL connection URL.")
    parsed_url = make_url(configured_url)
    if parsed_url.drivername not in {"postgresql", "postgresql+psycopg"}:
        raise ValueError("DATABASE_URL must use PostgreSQL with the psycopg driver.")
    return parsed_url.set(drivername="postgresql+psycopg")


def build_engine() -> Engine:
    return create_engine(
        get_database_url(), pool_pre_ping=True, connect_args={"options": "-c timezone=UTC"}
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
