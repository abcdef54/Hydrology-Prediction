"""Alembic entry point; weather pipeline implementation lives in research."""
from research.weather_forecast.migration import (
    revision, down_revision, branch_labels, depends_on, upgrade, downgrade,
)
