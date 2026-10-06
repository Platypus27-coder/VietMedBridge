from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.environment import environment_info, get_project_root, is_kaggle


def bootstrap(working_root=Path('/kaggle/working/vibiomir')):
    if not is_kaggle():
        raise RuntimeError('This bootstrap is for Kaggle. Use scripts/setup_env.ps1 or setup_env.sh locally.')
    if sys.version_info < (3, 11):
        raise RuntimeError('Python >=3.11 required. Use a compatible Kaggle runtime; do not create a venv.')
    print(json.dumps(environment_info(working_root), indent=2))
    missing = []
    for line in (get_project_root()/'requirements-kaggle.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, required = line.split('==')
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        if installed != required:
            missing.append(line)
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    for directory in ('data', 'reports', 'checkpoints', 'logs'):
        (working_root/directory).mkdir(parents=True, exist_ok=True)
    print('Kaggle dependencies ready; no virtual environment created.')


if __name__ == '__main__':
    bootstrap()
