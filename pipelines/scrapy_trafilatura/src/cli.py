from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import input_path, load_config, output_path, prepare_directories


def main(stage: str):
    parser = argparse.ArgumentParser(description=f'ViBioMIR {stage}')
    parser.add_argument('--config')
    parser.add_argument('--output-dir')
    parser.add_argument('--input')
    parser.add_argument('--query')
    parser.add_argument('--shard', default='0')
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--allow-pending', action='store_true')
    parser.add_argument('--pack-raw', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config, args.output_dir)
    if args.input:
        config['dataset']['links_path'] = str(Path(args.input).resolve())
    if args.query:
        config['dataset']['query_path'] = str(Path(args.query).resolve())
    if args.seed is not None:
        config['dataset']['sample_seed'] = args.seed
    from src.utils.config import config_hash
    config['_config_sha256'] = config_hash(config)
    prepare_directories(config)
    if stage == 'inventory':
        from src.inventory.build_inventory import build_inventory
        result = build_inventory(config)
    elif stage == 'shards':
        from src.inventory.build_shards import build_shards
        result = {'shards': [str(p) for p in build_shards(config)]}
    elif stage == 'crawl':
        from src.crawler.run import run_crawl
        shard = Path(args.shard) if not args.shard.isdigit() else output_path(config, f'data/crawl_shards/shard_{int(args.shard):05d}.parquet')
        result = run_crawl(config, shard, stop_after=args.stop_after)
    elif stage == 'extract':
        from src.extraction.worker import extract_raw
        result = extract_raw(config)
    elif stage == 'documents':
        from src.preprocessing.dedup import build_documents
        result = build_documents(config)
    elif stage == 'report':
        from src.reports.build_report import build_report
        result = build_report(config)
    elif stage == 'validate':
        from src.validation import validate_ingestion
        result = validate_ingestion(config, not args.allow_pending)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result['passed'] else 1)
    elif stage == 'stage_a':
        result = run_stage_a(config, args.sample_size)
    elif stage == 'pack':
        from src.crawler.raw_store import pack_raw
        result = {'packed_manifest': str(pack_raw(config, output_path(config, 'data/manifests/crawl_manifest.parquet')))}
    else:
        raise ValueError(stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_stage_a(config: dict, sample_size: int = 1000):
    if sample_size > 1000:
        raise ValueError('Stage A is capped at 1000 URLs. Stage B/full crawl requires reviewed gates and separate authorization.')
    from src.utils.environment import is_kaggle
    if is_kaggle():
        import urllib.request
        try:
            urllib.request.urlopen(config['environment']['network_probe_url'], timeout=10).close()
        except OSError as error:
            raise RuntimeError('Network unavailable. Use pre-downloaded raw dataset or enable allowed Internet access.') from error
    from src.inventory.build_inventory import build_inventory
    from src.inventory.build_shards import build_shards, sample_inventory
    from src.crawler.run import run_crawl
    from src.extraction.worker import extract_raw
    from src.preprocessing.dedup import build_documents
    from src.reports.build_report import build_report
    from src.validation import validate_ingestion
    print('Stage A: building/resuming immutable inventory', flush=True)
    build_inventory(config)
    print(f'Stage A: selecting at most {sample_size} URLs', flush=True)
    sample = sample_inventory(config, sample_size, config['dataset']['sample_seed'])
    for shard in build_shards(config, sample):
        print(f'Stage A: crawling {shard.name}', flush=True)
        run_crawl(config, shard)
    print('Stage A: offline extraction', flush=True)
    extract_raw(config)
    print('Stage A: canonical documents and complete source accounting', flush=True)
    build_documents(config)
    print('Stage A: reports and integrity checks', flush=True)
    report = build_report(config)
    integrity = validate_ingestion(config)
    if not integrity['passed']:
        raise RuntimeError(f'Integrity validation failed: {integrity["errors"]}')
    return report
