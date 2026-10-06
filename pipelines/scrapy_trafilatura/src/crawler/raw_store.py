from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import zstandard as zstd

from src.storage.io import atomic_bytes, atomic_json, parquet_rows, sha256_file, write_parquet
from src.utils.config import output_path


class RawStore:
    def __init__(self, config: dict):
        self.config = config

    def write(self, body: bytes, content_type: str) -> dict:
        checksum = hashlib.sha256(body).hexdigest()
        kind, extension = {'text/html': ('html', 'html.zst'), 'application/xhtml+xml': ('html', 'html.zst'),
                           'text/plain': ('text', 'txt.zst'), 'application/pdf': ('pdf', 'pdf')}[content_type]
        path = output_path(self.config, f'data/raw/{kind}/{checksum[:2]}/{checksum[2:4]}/{checksum}.{extension}')
        compressed = zstd.ZstdCompressor(level=self.config['raw_store']['compression_level']).compress(body) if extension.endswith('.zst') else body
        record = {'raw_path': str(path.relative_to(Path(self.config['_output_root']))), 'raw_sha256': checksum,
                  'raw_size_bytes': len(body), 'compressed_size_bytes': len(compressed),
                  'raw_member': None, 'raw_locator_version': 1}
        if path.exists():
            if self.read(record, len(body)) != body:
                raise ValueError(f'Existing raw blob differs from checksum: {path}')
        else:
            atomic_bytes(path, compressed)
        return record

    def read(self, record: dict, max_bytes: int) -> bytes:
        if record['raw_size_bytes'] > max_bytes:
            raise OverflowError('Raw input exceeds extraction size cap.')
        path = output_path(self.config, record['raw_path'])
        member = record.get('raw_member')
        if member:
            offset, size = record.get('raw_member_offset'), record.get('raw_member_size')
            if offset is not None and size is not None:
                if offset < 0 or size < 0 or size > max_bytes + 1024 * 1024 or offset + size > path.stat().st_size:
                    raise OverflowError('Packed member exceeds input size cap.')
                with path.open('rb') as handle:
                    handle.seek(offset)
                    compressed = handle.read(size)
            else:
                with tarfile.open(path, 'r:') as archive:
                    info = archive.getmember(member)
                    if info.size > max_bytes + 1024 * 1024:
                        raise OverflowError('Packed member exceeds input size cap.')
                    handle = archive.extractfile(info)
                    if handle is None:
                        raise ValueError('Raw archive member is not a file.')
                    compressed = handle.read(max_bytes + 1024 * 1024 + 1)
            stream = io.BytesIO(compressed)
            name = member
        else:
            stream = path.open('rb')
            name = path.name
        with stream:
            if name.endswith('.zst'):
                try:
                    with zstd.ZstdDecompressor().stream_reader(stream) as reader:
                        body = reader.read(max_bytes + 1)
                except zstd.ZstdError as error:
                    raise ValueError('Raw checksum/decompression failure.') from error
            else:
                body = stream.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise OverflowError('Decompressed input exceeds extraction size cap.')
        if len(body) != record['raw_size_bytes'] or hashlib.sha256(body).hexdigest() != record['raw_sha256']:
            raise ValueError('Raw checksum/size mismatch.')
        return body


def pack_raw(config: dict, manifest: Path) -> Path:
    """Publish immutable per-shard tar containers. Never delete loose recovery blobs."""
    import pyarrow.parquet as pq
    groups = {}
    for row in parquet_rows(manifest):
        if row['status'] == 'SUCCESS_RAW' and not row.get('raw_member'):
            groups.setdefault(row['shard_id'], {})[row['raw_path']] = row
    references = {}
    for shard, members in groups.items():
        identity = hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:16]
        path = output_path(config, f'data/raw/packed/{shard}-{identity}.tar')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        index = []
        with tarfile.open(temporary, 'w:') as archive:
            for relative, row in sorted(members.items()):
                source = output_path(config, relative)
                member = source.name
                RawStore(config).read(row, config['crawler']['max_response_bytes'])
                archive.add(source, arcname=member, recursive=False)
                references[relative] = (str(path.relative_to(Path(config['_output_root']))), member)
                index.append({'member': member, 'raw_sha256': row['raw_sha256']})
        temporary.replace(path)
        # Read headers once to publish actual data offsets (PAX headers may precede a member).
        with tarfile.open(path, 'r:') as archive:
            locations = {member.name: (member.offset_data, member.size) for member in archive}
        for entry in index:
            entry['offset'], entry['size'] = locations[entry['member']]
        for relative in members:
            container, member = references[relative]
            references[relative] = (container, member, *locations[member])
        atomic_json(path.with_suffix('.index.json'), {'sha256': sha256_file(path), 'members': index})
    def rows():
        for row in parquet_rows(manifest):
            if row.get('raw_path') in references:
                row['raw_path'], row['raw_member'], row['raw_member_offset'], row['raw_member_size'] = references[row['raw_path']]
                row['raw_locator_version'] = 2
            yield row
    target = manifest.with_name('crawl_manifest_packed.parquet')
    import pyarrow as pa
    packed_schema = pq.ParquetFile(manifest).schema_arrow
    for name in ('raw_member_offset', 'raw_member_size'):
        if name not in packed_schema.names:
            packed_schema = packed_schema.append(pa.field(name, pa.int64()))
    write_parquet(target, rows(), packed_schema)
    return target
