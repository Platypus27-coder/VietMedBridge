from pathlib import Path

import yaml

from src.utils.config import load_config


def test_effective_config_preserves_hash_when_reloaded(tmp_path):
    original = load_config(output_dir=tmp_path/'outputs')
    path = tmp_path/'effective.yaml'
    values = {k: v for k, v in original.items() if not k.startswith('_')}
    path.write_text(yaml.safe_dump(values), encoding='utf-8')
    reloaded = load_config(path)
    assert original['_config_sha256'] == reloaded['_config_sha256']
    assert Path(original['dataset']['links_path']).is_absolute()
