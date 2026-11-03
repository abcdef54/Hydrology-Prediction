"""Configuration and stable artifact identities for rerunnable research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, overrides=None):
    config = yaml.safe_load(Path(path).read_text())
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for name in ('work_dir', 'output_dir', 'fabdem', 'copernicus', 'boundary_file'):
        config[name] = str((ROOT / config[name]).resolve())
    for name in ('start', 'end', 'grouping_end'):
        if config.get(name):
            stamp = pd.Timestamp(config[name])
            if stamp.tzinfo is None:
                raise ValueError(f'{name} must include a timezone')
            config[name] = stamp.tz_convert('UTC').isoformat()
    if config.get('start') and config.get('end'):
        if pd.Timestamp(config['start']) >= pd.Timestamp(config['end']):
            raise ValueError('start must precede exclusive end')
    if config['sequence_length'] < 1 or config['min_pair_hours'] < 3:
        raise ValueError('Invalid sequence length or overlap requirement')
    if not config.get('radii_m') or min(config['radii_m']) <= 0:
        raise ValueError('Terrain radii must be positive metres')
    if config['boundary_buffer_m'] < max(config['radii_m']):
        raise ValueError('Boundary buffer must cover all terrain radii')
    return config


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)
