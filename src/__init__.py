"""Hydrology packages and compatibility for saved models using old imports."""

# Resolve legacy src.* imports without keeping AI implementations here.
from pathlib import Path

__path__.append(str(Path(__file__).parent / "ai"))
