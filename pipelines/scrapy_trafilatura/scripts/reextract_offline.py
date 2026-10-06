"""Re-extract a completed <=10k pilot in a separate root. Never invokes the crawler."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
import yaml

from src.extraction.worker import extract_raw
from src.pilot import reuse_inventory, save_deferred_urls
from src.preprocessing.dedup import build_documents
from src.reports.build_report import build_report
from src.storage.io import atomic_json, parquet_rows, sha256_file
from src.utils.config import load_config, output_path, prepare_directories
from src.validation import validate_ingestion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('outputs/stage_b_windows_10k'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/stage_b_windows_10k_reextract_v2'))
    args = parser.parse_args()
    source, target = args.source_root.resolve(), args.output_root.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Source and destination must be independent directories.')
    original_files = [source/name for name in ('data/manifests/crawl_manifest.parquet',
                      'data/manifests/extraction_manifest.parquet', 'data/processed/documents/documents.parquet',
                      'reports/stage_b/manual_review.csv')]
    originals = {str(p.relative_to(source)): sha256_file(p) for p in original_files}
    packed = source/'data/manifests/crawl_manifest_packed.parquet'
    crawl = list(parquet_rows(packed))
    if not 0 < len(crawl) <= 10000:
        raise ValueError('Offline rerun is limited to the existing <=10k pilot.')
    raw_count = sum(r['status'] == 'SUCCESS_RAW' for r in crawl)
    snapshot = json.loads((source/'checkpoints/inventory_state.json').read_text(encoding='utf-8'))['snapshot_id']
    config = load_config(output_dir=target)
    prepare_directories(config)
    state_path = output_path(config, 'checkpoints/offline_reextract_state.json')
    existing = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if existing and (existing['original_sha256'] != originals or existing['snapshot_id'] != snapshot):
        raise ValueError('Original inputs changed; use another destination.')
    state = {'status':'running', 'mode':'offline_reextract', 'source_root':str(source),
             'output_root':str(target), 'snapshot_id':snapshot, 'original_sha256':originals,
             'source_urls':len(crawl), 'raw_documents':raw_count, 'network_requests':0,
             'scale_authorized':False, 'started_at':datetime.now(timezone.utc).isoformat(), 'stages':{}}
    def perform(name, operation):
        state['current_stage'] = name
        atomic_json(state_path, state)
        print(f'Stage {name}: started', flush=True)
        started = time.monotonic()
        result = operation()
        state['stages'][name] = {'seconds':time.monotonic()-started, 'result':result}
        atomic_json(state_path, state)
        print(f'Stage {name}: complete ({state["stages"][name]["seconds"]:.1f}s)', flush=True)
        return result
    try:
        perform('inventory', lambda: reuse_inventory(config, source, snapshot))
        (target/'checkpoints/effective_config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        for relative in ('data/inventory/stage_b_sample.parquet', 'checkpoints/sample.json', 'requirements-lock.txt'):
            if (source/relative).exists():
                shutil.copyfile(source/relative, target/relative)
        # Copy only packed containers, with direct offsets retained in the manifest.
        for relative in sorted({r['raw_path'] for r in crawl if r['status'] == 'SUCCESS_RAW'}):
            origin, destination = (source/relative).resolve(), output_path(config, relative)
            if not origin.is_relative_to(source):
                raise ValueError('Source raw path escapes source root.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or sha256_file(destination) != sha256_file(origin):
                temporary = destination.with_suffix('.copy.tmp')
                shutil.copyfile(origin, temporary)
                temporary.replace(destination)
            if sha256_file(destination) != sha256_file(origin):
                raise ValueError('Packed container copy failed checksum verification.')
        shutil.copyfile(packed, target/'data/manifests/crawl_manifest.parquet')
        for relative in ('reports/stage_a/review_decision.json', 'reports/stage_b/coverage_decisions.json'):
            if (source/relative).exists():
                shutil.copyfile(source/relative, target/relative)
        perform('extraction', lambda: extract_raw(config))
        if pq.ParquetFile(target/'data/manifests/extraction_manifest.parquet').metadata.num_rows != raw_count:
            raise ValueError('Extraction does not account for every cached raw document.')
        perform('documents', lambda: build_documents(config))
        integrity = perform('integrity', lambda: validate_ingestion(config))
        if not integrity['passed']:
            raise ValueError(f'Integrity failed: {integrity["errors"][:5]}')
        report = perform('report', lambda: build_report(config, 'stage_b'))
        save_deferred_urls(config)
        report.update(mode='offline_reextract', source_root=str(source), network_requests=0,
                      next_stage={'100k_authorized':False, 'decision':'review_reextraction_quality'})
        atomic_json(target/'reports/stage_b/report.json', report)
        state['original_inputs_unchanged'] = all(sha256_file(source/name) == digest for name, digest in originals.items())
        if not state['original_inputs_unchanged']:
            raise ValueError('Baseline input checksums changed during rerun.')
        state.update(status='completed', current_stage=None, finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
