"""Run the observation worker with python -m src.backend.ingestion."""

from .worker import main


if __name__ == "__main__":
    raise SystemExit(main())
