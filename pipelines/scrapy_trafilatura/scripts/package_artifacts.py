from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, output_path


def package_artifacts(config, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in ('reports', 'logs', 'data/manifests', 'data/processed', 'data/mappings'):
            root = output_path(config, relative)
            for path in root.rglob('*'):
                if path.is_file() and path.suffix != '.sqlite':
                    archive.write(path, arcname=str(path.relative_to(Path(config['_output_root']))))
        for path in output_path(config, 'checkpoints').glob('*.json'):
            archive.write(path, arcname=f'checkpoints/{path.name}')
        effective = output_path(config, 'checkpoints/effective_config.yaml')
        archive.write(effective if effective.exists() else Path(config['_config_path']), 'config.yaml')
        for relative in ('data/source/dataset_manifest.json', 'data/inventory/stage_a_sample.parquet', 'data/inventory/stage_b_sample.parquet',
                         'data/inventory/gold_doc_map.parquet'):
            path = output_path(config, relative)
            if path.exists():
                archive.write(path, relative)
        lock = Path(config['_project_root'])/'requirements-lock.txt'
        if lock.exists():
            archive.write(lock, lock.name)
    temporary.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output-dir')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(package_artifacts(load_config(args.config, args.output_dir), args.destination))
