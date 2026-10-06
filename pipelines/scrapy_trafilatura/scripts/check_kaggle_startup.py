"""Read a bounded live-log snapshot without stopping the remote notebook."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--seconds', type=int, default=30)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 45:
        raise ValueError('Snapshot must be bounded to 1–45 seconds.')
    project = Path(__file__).resolve().parents[1]
    destination = args.destination or project/'outputs/kaggle_campaign/logs/startup.log'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as output:
        process = subprocess.Popen([sys.executable, str(project/'scripts/kaggle_remote.py'),
            'kernels', 'logs', args.kernel, '--follow'], stdout=output, stderr=subprocess.STDOUT, cwd=project)
        try:
            process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            process.terminate()  # Only this read-only local log viewer.
            process.wait(timeout=10)
    content = destination.read_text(encoding='utf-8', errors='replace')
    print(json.dumps({'log': str(destination), 'frontier_loaded': 'Selected frontier:' in content,
        'traceback_seen': 'Traceback (most recent call last)' in content,
        'dependencies_ready': 'Kaggle dependencies ready' in content,
        'progress_seen': '[ViBioMIR progress]' in content,
        'tqdm_seen': 'Crawl remaining' in content or 'Prepare frontier' in content}))
    print('\n'.join(content.splitlines()[-45:]))
