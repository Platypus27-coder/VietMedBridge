"""One bounded session; restored raw/events skip every committed URL outcome."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kaggle_batch import run_kaggle_batch
from src.utils.config import load_config
from src.utils.progress import BatchProgressMonitor, TqdmProgress


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--campaign-root', required=True, type=Path)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--compact-output', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    limits = config.get('kaggle_batch', {})
    with BatchProgressMonitor(config['_output_root'], args.shard_id,
            emit=TqdmProgress(),
            interval_seconds=limits.get('progress_interval_seconds', 30),
            stale_seconds=limits.get('progress_stale_seconds', 300)):
        report = run_kaggle_batch(config, args.campaign_root, args.shard_id,
                                  restore=args.restore, compact_output=args.compact_output)
    print(json.dumps({k: v for k, v in report.items() if k != 'domains'}, indent=2))
