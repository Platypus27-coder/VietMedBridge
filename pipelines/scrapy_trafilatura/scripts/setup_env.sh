#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_command="${PYTHON_EXECUTABLE:-python3.11}"
"$python_command" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
"$python_command" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${1:-requirements.txt}"
python --version
python -m pip --version
printf '%s\n' 'Ready. Activate with source .venv/bin/activate'
