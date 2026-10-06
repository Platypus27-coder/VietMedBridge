"""Prepare a private campaign Dataset and the first CPU notebook deployment."""
from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_source import package_source
from src.storage.io import atomic_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/kaggle_campaign'))
    args = parser.parse_args()
    manifest = json.loads((args.root/'input/campaign_manifest.json').read_text(encoding='utf-8'))
    dataset = f'{args.account}/vibiomir-crawl-{manifest["snapshot_id"][:8]}-20261004'
    kernel = f'{args.account}/vibiomir-crawl-shard-00000'
    package_source(args.root/'input/vibiomir_source.zip')
    atomic_json(args.root/'input/dataset-metadata.json', {
        'title': 'ViBioMIR frozen crawl campaign 2026-10-04', 'id': dataset,
        'licenses': [{'name': 'other'}],
        'description': 'Private execution inputs: native ViBioMIR snapshot, frozen URL frontiers, BTC ID mapping and reviewed ingestion code. Source: AIGuruTinix/ViBioMIR at immutable revision 0148f6f80ffafed5c005af6d506ccfd9d3fb47a7. No new public licensing grant.'})
    (args.root/'kernel').mkdir(parents=True, exist_ok=True)
    shutil.copyfile('notebooks/02_kaggle_crawl_shard.ipynb', args.root/'kernel/02_kaggle_crawl_shard.ipynb')
    atomic_json(args.root/'kernel/kernel-metadata.json', {
        'id': kernel, 'title': 'ViBioMIR crawl shard 00000',
        'code_file': '02_kaggle_crawl_shard.ipynb', 'language': 'python', 'kernel_type': 'notebook',
        'is_private': 'true', 'enable_gpu': 'false', 'enable_internet': 'true',
        'dataset_sources': [dataset], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
    atomic_json(args.root/'deployment.json', {'status': 'PREPARED', 'dataset': dataset, 'kernel': kernel,
        'snapshot_id': manifest['snapshot_id'], 'first_shard_urls': manifest['shards'][0]['rows'],
        'campaign_shards': manifest['shard_count'], 'execution': 'sequential', 'gpu': False,
        'dataset_private': True, 'notebook_private': True})
    print(json.dumps({'dataset': dataset, 'kernel': kernel,
                      'input_bytes': sum(p.stat().st_size for p in (args.root/'input').iterdir() if p.is_file())}))
