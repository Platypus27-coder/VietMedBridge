"""Download the two native Parquet files at one immutable Hugging Face revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage.io import atomic_json, sha256_file
from src.utils.config import load_config, input_path, output_path, prepare_directories


def fetch(config: dict, repository='AIGuruTinix/ViBioMIR', revision='main') -> dict:
    metadata_url = f'https://huggingface.co/api/datasets/{repository}/revision/{revision}'
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        metadata = json.load(response)
    commit = metadata['sha']
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repository}/tree/{commit}?recursive=true', timeout=30) as response:
        files = {row['path']: row for row in json.load(response)}
    prepare_directories(config)
    fetched = []
    for name, key in [('links_corpus.parquet', 'links_path'), ('query.parquet', 'query_path')]:
        path = input_path(config, key)
        row = files[name]
        expected = row.get('lfs', {}).get('oid')
        if path.exists():
            if expected and sha256_file(path) != expected:
                raise ValueError(f'Existing {path} differs from the pinned HF source; will not overwrite.')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.download')
            with urllib.request.urlopen(f'https://huggingface.co/datasets/{repository}/resolve/{commit}/{name}', timeout=120) as response, temporary.open('wb') as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != row['size'] or (expected and sha256_file(temporary) != expected):
                raise ValueError(f'Download checksum/size mismatch: {name}')
            temporary.replace(path)
        fetched.append({'name': name, 'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size})
    result = {'repository': repository, 'revision': commit, 'files': fetched}
    atomic_json(output_path(config, 'data/source/huggingface_source.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--repo', default='AIGuruTinix/ViBioMIR')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    print(json.dumps(fetch(load_config(args.config), args.repo, args.revision), indent=2))
