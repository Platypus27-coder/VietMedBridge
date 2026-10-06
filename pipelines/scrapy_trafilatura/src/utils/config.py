from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .environment import get_project_root, get_writable_root, is_kaggle


def load_config(path: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or get_project_root() / 'configs/config.yaml').resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = get_project_root()
    configured = config['environment']['output_dir']
    output = Path(output_dir) if output_dir is not None else (get_writable_root() if is_kaggle() and configured == '.' else Path(configured))
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if is_kaggle() and not output.is_relative_to(Path('/kaggle/working')):
        raise ValueError('Kaggle outputs must be under /kaggle/working; /kaggle/input is read-only.')
    if config['inventory']['remove_tracking_params']:
        raise ValueError('Removing query parameters is not supported in normalization v1.')
    if 429 in config['crawler']['retry_http_codes']:
        raise ValueError('429 requires the durable cooldown queue, not immediate RetryMiddleware retries.')
    if config['raw_store']['layout'] not in ('loose_files', 'packed_shards'):
        raise ValueError('Unsupported raw layout.')
    config['environment']['output_dir'] = str(output)
    for key in ('links_path', 'query_path'):
        source = Path(config['dataset'][key])
        config['dataset'][key] = str((source if source.is_absolute() else root / source).resolve())
    config['_project_root'] = str(root)
    config['_output_root'] = str(output)
    config['_config_path'] = str(path)
    config['_config_sha256'] = config_hash(config)
    return config


def config_hash(config: dict) -> str:
    effective = {k: v for k, v in config.items() if not k.startswith('_')}
    return hashlib.sha256(json.dumps(effective, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def output_path(config: dict, relative: str) -> Path:
    path = (Path(config['_output_root']) / relative).resolve()
    if not path.is_relative_to(Path(config['_output_root'])):
        raise ValueError(f'Output escapes writable root: {relative}')
    return path


def input_path(config: dict, key: str) -> Path:
    path = Path(config['dataset'][key])
    return path if path.is_absolute() else Path(config['_project_root']) / path


def prepare_directories(config: dict) -> None:
    for directory in ('data/source', 'data/inventory', 'data/crawl_shards', 'data/raw',
                      'data/manifests/crawl', 'data/manifests/extraction', 'data/processed/extracted',
                      'data/processed/documents', 'data/mappings', 'crawl_jobs', 'checkpoints',
                      'reports/stage_a', 'reports/stage_b', 'logs'):
        output_path(config, directory).mkdir(parents=True, exist_ok=True)
