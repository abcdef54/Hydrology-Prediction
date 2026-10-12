"""Migrations use DATABASE_URL; no secrets stored in Alembic configuration."""

from alembic import context
import sqlalchemy as sa

from src.backend.db.models import Base
from src.backend.db.session import build_engine, get_database_url


def include_name(name, type_, parent_names):
    # Autogeneration must never propose changes to unrelated/public tables.
    return type_ != "schema" or name == "hydrology"


options = dict(
    target_metadata=Base.metadata,
    include_schemas=True,
    include_name=include_name,
    version_table="hydrology_alembic_version",
    version_table_schema="public",
    compare_type=True,
    compare_server_default=True,
    transaction_per_migration=False,
)

if context.is_offline_mode():
    context.configure(url=get_database_url(), literal_binds=True, **options)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = build_engine()
    try:
        with engine.begin() as connection:
            connection.execute(sa.text("SELECT pg_advisory_xact_lock(20261012, 1)"))
            context.configure(connection=connection, **options)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()
