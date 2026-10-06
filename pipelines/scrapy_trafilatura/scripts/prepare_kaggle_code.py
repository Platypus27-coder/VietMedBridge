"""Prepare a small, separately versioned runner Dataset without reuploading data."""
from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_source import package_source
from src.storage.io import atomic_json, sha256_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/kaggle_campaign'))
    args = parser.parse_args()
    deployment = json.loads((args.root/'deployment.json').read_text(encoding='utf-8'))
    account = deployment['kernel'].split('/')[0]
    code_dataset = f'{account}/vibiomir-runner-20261004'
    code_root = args.root/'code_input'
    archive = package_source(code_root/'vibiomir_source.zip')
    atomic_json(code_root/'source_package_manifest.json', {
        'type': 'vibiomir_pipeline_code', 'archive': archive.name, 'sha256': sha256_file(archive),
        'baseline': 'Scrapy 2.13.4; Trafilatura 2.1.0; Python >=3.11',
        'paired_native_extraction_checks': 18, 'tests_full_suite_passed': 51})
    atomic_json(code_root/'dataset-metadata.json', {
        'id': code_dataset, 'title': 'ViBioMIR reviewed Kaggle runner 2026-10-04',
        'licenses': [{'name': 'other'}], 'description': 'Private execution code, versioned separately from immutable crawl frontiers.'})
    metadata_path = args.root/'kernel/kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['dataset_sources'] = [deployment['dataset'], code_dataset]
    atomic_json(metadata_path, metadata)
    shutil.copyfile('notebooks/02_kaggle_crawl_shard.ipynb', args.root/'kernel/02_kaggle_crawl_shard.ipynb')
    deployment.update(runner_dataset=code_dataset, source_sha256=sha256_file(archive))
    atomic_json(args.root/'deployment.json', deployment)
    print(json.dumps({'runner_dataset': code_dataset, 'source_sha256': deployment['source_sha256'], 'code_bytes': archive.stat().st_size}))
