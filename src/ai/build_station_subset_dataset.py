"""Build station subsets from a frozen regional pool, preserving exact labels/splits."""
import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from src.ai.regional_data.builder import build_dataset, sha, write_json
from src.ai.regional_data.validation import validate_dataset


def build_station_subset(source, output, station_ids):
    source, output = Path(source).resolve(), Path(output).resolve()
    ids = list(station_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Select at least one station, without duplicates')
    mapping = json.loads((source/'station_mapping.json').read_text())
    if set(ids)-set(mapping):
        raise ValueError(f'Unknown station IDs: {sorted(set(ids)-set(mapping))}')
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['status'] != 'complete' or json.loads((source/'validation.json').read_text())['status'] != 'PASS':
        raise ValueError('Source must be a completed, validated dataset')
    for name, expected in manifest['artifact_sha256'].items():
        if sha(source/name) != expected:
            raise ValueError(f'Source artifact changed: {name}')
    selected = {sid: dict(copy.deepcopy(mapping[sid]), station_index=i) for i, sid in enumerate(ids)}
    config = json.loads((source/'config.json').read_text())
    config.update(expected_station_count=len(ids), require_all_eda_groups=False,
        station_selection_description=f'Explicit station subset of {source.name}; source targets and split memberships preserved.')
    def grid(name):
        return pd.read_parquet(source/name).set_index('timestamp_utc')
    paths = [source/'manifest.json', *[source/name for name in manifest['artifact_sha256']]]
    paths.append(Path(__file__).resolve())
    clusters = pd.read_csv(source/'eda1_station_groups.csv', dtype={'station_id': str})
    build_dataset(output, level=grid('water.parquet')[ids],
        rain_standard=grid('rainfall_hourly_standard.parquet'), rain_strict=grid('rainfall_hourly_strict.parquet'),
        weather=grid('weather.parquet'), stations=selected,
        clusters=clusters[clusters.station_id.isin(ids)], config=config, source_paths=paths,
        rain_catalog=pd.read_parquet(source/'rain_station_catalog.parquet'))
    original = pd.read_parquet(source/'samples.parquet')
    expected = original[original.station_id.isin(ids)].copy()
    actual = pd.read_parquet(output/'samples.parquet')
    columns = [c for c in original.columns if c not in ('sample_id', 'station_index')]
    sort = ['station_id', 'anchor_utc']
    pd.testing.assert_frame_equal(expected[columns].sort_values(sort).reset_index(drop=True),
                                  actual[columns].sort_values(sort).reset_index(drop=True))
    for fold, stats in json.loads((output/'normalizers.json').read_text()).items():
        old = json.loads((source/'normalizers.json').read_text())[fold]
        for sid in stats['stations']:
            if stats['stations'][sid] != old['stations'][sid]:
                raise AssertionError(f'Station normalization changed: {fold}/{sid}')
    write_json(output/'station_subset.json', dict(source_dataset=str(source),
        source_manifest_sha256=sha(source/'manifest.json'), station_ids=ids,
        sample_parity='PASS; all original sample fields except new sample/station indices identical',
        station_normalizer_parity='PASS', shared_normalizers='Refitted on selected training population only'))
    with (output/'REPORT.md').open('a') as report:
        report.write('\n## Station subset\n\nSelected station IDs: '+', '.join(ids)+'. '
            'Sample targets, anchors, missingness and split memberships match the parent dataset. '
            'Station indices are remapped; use newly trained checkpoints. Shared scalers are refitted on train only.\n')
    return validate_dataset(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--station-ids', nargs='+', required=True)
    args = parser.parse_args()
    print(json.dumps(build_station_subset(args.source_dir, args.output_dir, args.station_ids), indent=2))


if __name__ == '__main__':
    main()
