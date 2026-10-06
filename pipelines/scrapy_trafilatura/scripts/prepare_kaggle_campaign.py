"""Export frozen, bounded frontiers without copying the full working inventory."""
from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from src.storage.io import atomic_json, sha256_file


def prepare_campaign(source_root: Path, destination: Path, urls_per_shard=100000):
    if not 0 < urls_per_shard <= 100000:
        raise ValueError('Kaggle frontiers must contain at most 100,000 URLs.')
    checkpoint = destination/'campaign_manifest.json'
    if checkpoint.exists():
        manifest = json.loads(checkpoint.read_text(encoding='utf-8'))
        snapshot = json.loads((source_root/'data/source/dataset_manifest.json').read_text(encoding='utf-8'))
        if manifest['snapshot_id'] != snapshot['snapshot_id'] or manifest['urls_per_shard'] != urls_per_shard:
            raise ValueError('Campaign snapshot/partition changed; use a new destination.')
        for item in manifest['files']:
            if sha256_file(destination/item['name']) != item['sha256']:
                raise ValueError(f'Campaign checksum mismatch: {item["name"]}')
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(source_root/'data/inventory/unique_urls.parquet')
    snapshot = json.loads((source_root/'data/source/dataset_manifest.json').read_text(encoding='utf-8'))
    shards, writer, count, total = [], None, 0, 0
    try:
        for batch in source.iter_batches(batch_size=10000):
            table = pa.Table.from_batches([batch])
            offset = 0
            while offset < table.num_rows:
                if writer is None:
                    name = f'shard_{len(shards):05d}.parquet'
                    path = destination/name
                    writer = pq.ParquetWriter(path.with_suffix('.tmp'), source.schema_arrow, compression='zstd')
                    count = 0
                length = min(urls_per_shard-count, table.num_rows-offset)
                writer.write_table(table.slice(offset, length))
                offset += length
                count += length
                total += length
                if count == urls_per_shard:
                    writer.close()
                    writer = None
                    path.with_suffix('.tmp').replace(path)
                    shards.append({'name': name, 'rows': count, 'sha256': sha256_file(path)})
                    print(f'Prepared {name}: {count:,} URLs', flush=True)
        if writer is not None:
            writer.close()
            writer = None
            path.with_suffix('.tmp').replace(path)
            shards.append({'name': name, 'rows': count, 'sha256': sha256_file(path)})
    finally:
        if writer is not None:
            writer.close()
    if total != source.metadata.num_rows or sum(s['rows'] for s in shards) != total:
        raise ValueError('Campaign partition lost source URLs.')
    copied = []
    for source_info in snapshot['sources']:
        source_path = Path(source_info['path'])
        if not source_path.exists():
            source_path = source_root/'data/source'/Path(source_info['original_path']).name
        target = destination/source_path.name
        shutil.copyfile(source_path, target)
        if sha256_file(target) != source_info['sha256']:
            raise ValueError('Dataset snapshot checksum mismatch.')
        copied.append({'name': target.name, 'sha256': source_info['sha256']})
    for name in ('url_doc_map.parquet', 'invalid_urls.parquet'):
        target = destination/name
        shutil.copyfile(source_root/'data/inventory'/name, target)
        copied.append({'name': name, 'sha256': sha256_file(target)})
    atomic_json(destination/'dataset_manifest.json', snapshot)
    copied.append({'name': 'dataset_manifest.json', 'sha256': sha256_file(destination/'dataset_manifest.json')})
    manifest = {'format_version': 1, 'snapshot_id': snapshot['snapshot_id'],
                'source_rows': snapshot['sources'][0]['rows'], 'unique_urls': total,
                'urls_per_shard': urls_per_shard, 'shard_count': len(shards),
                'partition': 'frozen_inventory_order_contiguous_ranges',
                'execution': 'sequential', 'shards': shards, 'files': shards+copied,
                'gold_coverage_policy': 'proceed_without_qrels',
                'failure_policy': 'retain_for_later', 'global_dedup': 'after_all_shards'}
    atomic_json(checkpoint, manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('outputs/stage_a_native'))
    parser.add_argument('--destination', type=Path, default=Path('outputs/kaggle_campaign/input'))
    args = parser.parse_args()
    result = prepare_campaign(args.source_root, args.destination)
    print(json.dumps({k: result[k] for k in ('snapshot_id', 'source_rows', 'unique_urls', 'shard_count')}))
