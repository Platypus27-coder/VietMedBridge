"""Prepare a private continuation using pinned notebook output, not local raw."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_source import package_source
from src.storage.io import atomic_json, sha256_file


def prepare_resume(campaign_root, destination, tag, reuse_runner=False):
    previous = json.loads((campaign_root/'deployment.json').read_text(encoding='utf-8'))
    if previous['status'] != 'COMPLETE':
        raise ValueError('Previous notebook must stop before a sequential continuation starts.')
    report = json.loads(Path(previous['report']).read_text(encoding='utf-8'))
    if report['status'] != 'PARTIAL' or not report['crawl_remaining']:
        raise ValueError('Continuation requires a partial shard with pending URLs.')
    if report['snapshot_id'] != previous['snapshot_id'] or not report['bundle']['verified'] or not report['integrity_passed']:
        raise ValueError('Previous report has an unverified bundle or different snapshot.')
    if report['shard_id'] != previous['shard_id']:
        raise ValueError('Previous report refers to a different shard.')
    checksum = Path(previous['report']).with_name(f"vibiomir_shard_{report['shard_id']:05d}.tar.sha256")
    if checksum.read_text(encoding='utf-8').split()[0] != report['bundle']['sha256']:
        raise ValueError('Previous report and checkpoint checksum disagree.')
    if (destination/'submission.json').exists():
        raise ValueError('This continuation has already been submitted; inspect its status before another launch.')
    atomic_json(destination/'previous_deployment.json', previous)
    account = previous['kernel'].split('/')[0]
    shard_id = report['shard_id']
    kernel = f'{account}/vibiomir-crawl-shard-{shard_id:05d}-{tag}'
    runner = previous['runner_dataset'] if reuse_runner else f'{account}/vibiomir-runner-{tag}'
    kernel_root, code_root = destination/'kernel', destination/'code_input'
    kernel_root.mkdir(parents=True, exist_ok=True)
    if not reuse_runner:
        code_root.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(Path('notebooks/02_kaggle_crawl_shard.ipynb').read_text(encoding='utf-8'))
    parameters = ''.join(notebook['cells'][0]['source'])
    for before, after in (
        ('SHARD_ID = 0', f'SHARD_ID = {shard_id}'),
        ('REQUIRE_RESTORE = False', 'REQUIRE_RESTORE = True'),
        ('EXPECTED_RESTORE_SHA256 = None', f"EXPECTED_RESTORE_SHA256 = {report['bundle']['sha256']!r}"),
        ('MAX_EXTRACTION_EVENT_GIB = 1', 'MAX_EXTRACTION_EVENT_GIB = 2'),
    ):
        if parameters.count(before) != 1:
            raise ValueError(f'Unexpected notebook parameter: {before}')
        parameters = parameters.replace(before, after)
    notebook['cells'][0]['source'] = parameters.splitlines(keepends=True)
    atomic_json(kernel_root/'02_kaggle_crawl_shard.ipynb', notebook)
    output_source = f"{previous['kernel']}/{previous['kernel_version']}"
    atomic_json(kernel_root/'kernel-metadata.json', {
        'id': kernel, 'title': kernel.split('/')[1].replace('-', ' '),
        'code_file': '02_kaggle_crawl_shard.ipynb', 'language': 'python',
        'kernel_type': 'notebook', 'is_private': 'true', 'enable_gpu': 'false',
        'enable_internet': 'true', 'dataset_sources': [previous['dataset'], runner],
        'kernel_sources': [output_source], 'competition_sources': [], 'model_sources': []})
    if reuse_runner:
        source_hash = previous['source_sha256']
    else:
        archive = package_source(code_root/'vibiomir_source.zip')
        source_hash = sha256_file(archive)
        atomic_json(code_root/'source_package_manifest.json', {
            'type': 'vibiomir_pipeline_code', 'archive': archive.name, 'sha256': source_hash,
            'features': ['tqdm', '30-second telemetry', 'required checkpoint restore'],
            'baseline': 'Scrapy 2.13.4; Trafilatura 2.1.0; Python >=3.11'})
        atomic_json(code_root/'dataset-metadata.json', {
            'id': runner, 'title': f'ViBioMIR runner {tag}', 'licenses': [{'name': 'other'}],
            'description': 'Private reviewed continuation code with tqdm and fail-closed checkpoint restore.'})
    plan = {'status': 'PREPARED', 'kernel': kernel, 'kernel_url': f'https://www.kaggle.com/code/{kernel}',
            'dataset': previous['dataset'], 'runner_dataset': runner, 'source_sha256': source_hash,
            'previous_kernel': previous['kernel'], 'previous_version': previous['kernel_version'],
            'previous_output_source': output_source, 'previous_report': previous['report'],
            'expected_restore_sha256': report['bundle']['sha256'],
            'snapshot_id': previous['snapshot_id'], 'shard_id': shard_id,
            'frontier_urls': report['frontier_urls'], 'pending_urls': report['crawl_remaining'],
            'expected_skipped_urls': report['frontier_urls']-report['crawl_remaining'],
            'require_restore': True, 'tqdm': True, 'progress_interval_seconds': 30,
            'failure_policy': 'retain_for_later', 'gpu': False, 'local_raw_download_required': False,
            'extraction_event_budget_gib': 2, 'reviewed_runner_reused': reuse_runner,
            'campaign_shards': report['campaign_shards'],
            'campaign_unique_urls': report['campaign_unique_urls'], 'execution': 'sequential',
            'completed_shards': previous.get('completed_shards', []),
            'auto_restore': True, 'session_budget_version': 1,
            'session_hours': 11, 'crawl_fixed_hours': None, 'extraction_fixed_hours': None,
            'extraction_reserve_minutes': 60, 'export_reserve_minutes': 30,
            'kernel_timeout_seconds': 43200}
    atomic_json(destination/'deployment_plan.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, default=Path('outputs/kaggle_campaign'))
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--reuse-runner', action='store_true', help='Reuse the previous reviewed code Dataset unchanged.')
    args = parser.parse_args()
    print(json.dumps(prepare_resume(args.campaign_root, args.destination, args.tag, args.reuse_runner), indent=2))
