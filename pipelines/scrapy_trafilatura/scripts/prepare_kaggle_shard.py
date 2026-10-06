"""Prepare one fresh frozen shard with a reused or separately versioned runner."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage.io import atomic_json, sha256_file
from scripts.package_source import package_source


def prepare_shard(campaign_root, destination, shard_id, runner_tag=None):
    previous = json.loads((campaign_root/'deployment.json').read_text(encoding='utf-8'))
    if previous['status'] != 'COMPLETE':
        raise ValueError('Previous notebook must stop before another sequential shard starts.')
    manifest = json.loads((campaign_root/'input/campaign_manifest.json').read_text(encoding='utf-8'))
    if previous['snapshot_id'] != manifest['snapshot_id']:
        raise ValueError('Dataset snapshot differs from the previous deployment.')
    if not 0 <= shard_id < manifest['shard_count']:
        raise ValueError('Shard ID is outside the frozen campaign.')
    if shard_id in previous.get('completed_shards', []):
        raise ValueError('Selected shard has already completed; refusing a fresh duplicate crawl.')
    if (destination/'submission.json').exists():
        raise ValueError('This shard has already been submitted; inspect its status before relaunching.')
    item = manifest['shards'][shard_id]
    if item['rows'] > 100000 or item['name'] != f'shard_{shard_id:05d}.parquet':
        raise ValueError('Invalid/oversized frozen frontier.')
    if sha256_file(campaign_root/'input'/item['name']) != item['sha256']:
        raise ValueError('Selected frozen frontier checksum mismatch.')
    account = previous['kernel'].split('/')[0]
    kernel = f'{account}/vibiomir-crawl-shard-{shard_id:05d}'
    runner = f'{account}/vibiomir-runner-{runner_tag}' if runner_tag else previous['runner_dataset']
    notebook = json.loads(Path('notebooks/02_kaggle_crawl_shard.ipynb').read_text(encoding='utf-8'))
    parameters = ''.join(notebook['cells'][0]['source'])
    for before, after in (
        ('SHARD_ID = 0', f'SHARD_ID = {shard_id}'),
        ('AUTO_RESTORE = True', 'AUTO_RESTORE = False'),
        ('MAX_EXTRACTION_EVENT_GIB = 1', 'MAX_EXTRACTION_EVENT_GIB = 2'),
    ):
        if parameters.count(before) != 1:
            raise ValueError(f'Unexpected notebook parameter: {before}')
        parameters = parameters.replace(before, after)
    notebook['cells'][0]['source'] = parameters.splitlines(keepends=True)
    kernel_root = destination/'kernel'
    kernel_root.mkdir(parents=True, exist_ok=True)
    atomic_json(destination/'previous_deployment.json', previous)
    atomic_json(kernel_root/'02_kaggle_crawl_shard.ipynb', notebook)
    atomic_json(kernel_root/'kernel-metadata.json', {
        'id': kernel, 'title': f'ViBioMIR crawl shard {shard_id:05d}',
        'code_file': '02_kaggle_crawl_shard.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': 'true', 'enable_gpu': 'false',
        'enable_internet': 'true', 'dataset_sources': [previous['dataset'], runner],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []})
    if runner_tag:
        code_root = destination/'code_input'
        archive = package_source(code_root/'vibiomir_source.zip')
        source_hash = sha256_file(archive)
        atomic_json(code_root/'source_package_manifest.json', {
            'type': 'vibiomir_pipeline_code', 'archive': archive.name, 'sha256': source_hash,
            'features': ['tqdm', '30-second telemetry', 'required checkpoint restore', 'shared session deadline'],
            'baseline': 'Scrapy 2.13.4; Trafilatura 2.1.0; Python >=3.11'})
        atomic_json(code_root/'dataset-metadata.json', {
            'id': runner, 'title': f'ViBioMIR runner {runner_tag}', 'licenses': [{'name': 'other'}],
            'description': 'Private execution code with one session deadline and extraction/export reserves.'})
    else:
        source_hash = previous['source_sha256']
    plan = {'status': 'PREPARED', 'kernel': kernel, 'kernel_url': f'https://www.kaggle.com/code/{kernel}',
            'dataset': previous['dataset'], 'runner_dataset': runner,
            'source_sha256': source_hash, 'snapshot_id': manifest['snapshot_id'],
            'shard_id': shard_id, 'ordinal_shard': shard_id+1, 'frontier_urls': item['rows'],
            'frontier_sha256': item['sha256'], 'frontier_name': item['name'],
            'pending_urls': item['rows'], 'campaign_shards': manifest['shard_count'],
            'campaign_unique_urls': manifest['unique_urls'], 'execution': 'sequential',
            'completed_shards': previous.get('completed_shards', []),
            'tqdm': True, 'progress_interval_seconds': 30, 'failure_policy': 'retain_for_later',
            'gpu': False, 'require_restore': False, 'auto_restore': False,
            'local_raw_download_required': False, 'extraction_event_budget_gib': 2,
            'reviewed_runner_reused': not bool(runner_tag), 'session_budget_version': 1,
            'session_hours': 11, 'crawl_fixed_hours': None, 'extraction_fixed_hours': None,
            'extraction_reserve_minutes': 60, 'export_reserve_minutes': 30,
            'kernel_timeout_seconds': 43200,
            'previous_kernel': previous['kernel'], 'previous_report': previous['report']}
    atomic_json(destination/'deployment_plan.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, default=Path('outputs/kaggle_campaign'))
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--shard-id', required=True, type=int)
    parser.add_argument('--runner-tag', help='Package updated code in a new private runner Dataset.')
    args = parser.parse_args()
    print(json.dumps(prepare_shard(args.campaign_root, args.destination, args.shard_id, args.runner_tag), indent=2))
