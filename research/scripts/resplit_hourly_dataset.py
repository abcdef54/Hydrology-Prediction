"""Create a separate, audited chronological split from an existing hourly snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.scripts.build_training_dataset_1h_top4 import audit


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-start', required=True)
    parser.add_argument('--test-start', required=True)
    args = parser.parse_args()
    val_start, test_start = pd.Timestamp(args.validation_start), pd.Timestamp(args.test_start)
    if val_start.tzinfo is None or test_start.tzinfo is None:
        parser.error('Split times must include a timezone')
    table = pd.read_csv(args.source_dir / 'hourly.csv')
    table['timestamp_utc'] = pd.to_datetime(table.timestamp_utc, utc=True)
    table = table.set_index('timestamp_utc')
    source_manifest = json.loads((args.source_dir / 'manifest.json').read_text())
    targets = source_manifest['target_columns']
    horizon = max(int(col.rsplit('_', 1)[-1].removesuffix('m')) for col in targets)
    purge = pd.Timedelta(minutes=horizon)
    if not table.index.min() < val_start < test_start <= table.index.max():
        parser.error('Split boundaries must fall inside the hourly data range')
    splits = {
        'train': table.loc[table.index + purge < val_start],
        'val': table.loc[(table.index >= val_start) & (table.index + purge < test_start)],
        'test': table.loc[table.index >= test_start],
    }
    checks = audit(table, splits)
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory; existing datasets are preserved')
    args.output_dir.mkdir(parents=True)
    for name in ['hourly.csv', 'missingness.csv']:
        shutil.copyfile(args.source_dir / name, args.output_dir / name)
    summary = {}
    for name, frame in splits.items():
        frame.to_csv(args.output_dir / f'{name}.csv', index=True)
        summary[name] = {
            'rows': len(frame),
            'complete_targets': int(frame[targets].notna().all(axis=1).sum()),
            'start_utc': str(frame.index.min()), 'end_utc': str(frame.index.max()),
            'start_vietnam': str(frame.index.min().tz_convert('Asia/Ho_Chi_Minh')),
            'end_vietnam': str(frame.index.max().tz_convert('Asia/Ho_Chi_Minh')),
        }
    manifest = dict(source_manifest)
    manifest.update(
        split_created_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_dataset=str(args.source_dir.resolve()),
        parent_manifest_sha256=checksum(args.source_dir / 'manifest.json'),
        split_rule='Explicit chronological boundaries; purge maximum forecast horizon before validation/test',
        validation_start=str(val_start), test_start=str(test_start), purge_minutes=horizon,
        splits=summary, audit_checks=checks,
        outputs_sha256={path.name: checksum(path) for path in args.output_dir.glob('*.csv')},
    )
    assert checksum(args.source_dir / 'hourly.csv') == checksum(args.output_dir / 'hourly.csv')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    lines = ['# Dataset Dã Viên: validation từ 01/08 đến 15/09', '',
        'Nguồn: cùng snapshot PostgreSQL của dataset cha; không đổi feature, target hoặc dữ liệu tổng.', '',
        f'- Dataset cha: `{args.source_dir}`.',
        f'- Validation bắt đầu: {val_start}; test thăm dò bắt đầu: {test_start}.',
        f'- Purge {horizon} phút trước mỗi ranh giới; giữ NaN target, không điền nhãn tương lai.',
        '- Test đã được phân tích trước đó, không phải tập cuối chưa từng xem.',
        '- Số hàng dưới đây chưa trừ warmup 95 bước của chuỗi 96 giờ.', '',
        '| Tập | Mốc đầu (VN) | Mốc cuối (VN) | Hàng | Đủ 5 target |',
        '|---|---|---|---:|---:|']
    for name, row in summary.items():
        lines.append(f'| {name} | {row["start_vietnam"]} | {row["end_vietnam"]} | {row["rows"]} | {row["complete_targets"]} |')
    lines += ['', f'Audit: {len(checks)} kiểm tra đạt. `hourly.csv` giữ nguyên SHA-256 so với dataset cha.', '',
        'Train và val bị loại 24 mốc giờ ở cuối mỗi tập trước ranh giới. Đợt mưa/lũ liên tục có thể vẫn tạo phụ thuộc giữa các tập; purge chỉ bảo đảm target không vượt ranh giới.']
    (args.output_dir / 'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Audit: {len(checks)} passed. Hourly source is byte-identical.')


if __name__ == '__main__':
    main()
