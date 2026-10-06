"""Read live progress without compacting or repairing an active JSONL writer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage.io import snapshot_binary_reader


def status(root: Path):
    latest = {}
    for path in sorted((root/'data/manifests/crawl').glob('part-*.jsonl*')):
        if not (path.name.endswith('.jsonl') or path.name.endswith('.jsonl.open')):
            continue
        with snapshot_binary_reader(path) as handle:
            for line in handle:
                if not line.endswith(b'\n'):
                    break
                record = json.loads(line)
                old = latest.get(record['crawl_url_id'])
                if old is None or (record['attempt_no'], record['event_seq']) > (old['attempt_no'], old['event_seq']):
                    latest[record['crawl_url_id']] = record
    result = {'crawl_completed': len(latest), 'crawl': dict(Counter(r['status'] for r in latest.values()))}
    for name in ('run_state', 'pilot_resources', 'supervisor_state', 'extraction_state'):
        path = root/f'checkpoints/{name}.json'
        if path.exists():
            with snapshot_binary_reader(path) as handle:
                snapshot = json.load(handle)
            result[name] = {k: v for k, v in snapshot.items() if k != 'observations'}
            if snapshot.get('observations'):
                result[name]['current_stage'] = snapshot['observations'][-1]['stage']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/stage_b_windows_10k'))
    print(json.dumps(status(parser.parse_args().output_dir), indent=2))
