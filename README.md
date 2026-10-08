# Hydrology workspace

- `src/ai/`: model training and evaluation code.
- `scripts/`: hourly top-four-rain dataset builder, MLflow metrics notebook, and seasonal analysis notebook.
- `data/`: original observations, metadata, SQLite master, and provenance/checksums.
- `train_data/1h/` and `train_data/1h_top4/`: train/validation/test splits.
- `reports/`: current hourly datasets and their audit/missingness metadata.
- `mlflow.db`, `mlartifacts/`, `model/`, `pretrained/`: experiment history and model weights.
- `archive/2026-10-07/`: earlier utilities, GIS inputs, documents, and analysis exports.

Run MLflow from this directory:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000. Training uses this local tracking server.

For the current dataset, pass `--horizon 1h --dataset-dir train_data/1h_top4`
to `.venv/bin/python src/ai/train.py`; use `--help` for the other required options.

Open `scripts/mlflow_metrics.ipynb` to evaluate saved runs. New exports go to
`reports/mlflow_metrics/`. Open `scripts/seasonal_analysis.ipynb` for rainfall
and water-level analysis.

The archive preserves original relative paths. Restore the required files to
those paths before running archived utilities; include their report/data
inputs and sibling imports. `archive/2026-10-07/cleanup_manifest.json` lists
every moved/deleted file, its reason, and its original SHA-256 checksum.
