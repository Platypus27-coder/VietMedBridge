from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pilot import run_stage_b
from src.utils.config import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reviewed Stage B pilot, capped at 10,000 URLs.')
    parser.add_argument('--config')
    parser.add_argument('--output-dir', default='outputs/stage_b_windows_10k')
    parser.add_argument('--stage-a-root', type=Path, default=Path('outputs/stage_a_native'))
    parser.add_argument('--sample-size', type=int, default=10000)
    parser.add_argument('--background-supervisor', action='store_true', help='Watch the existing Windows pilot, recover a failed runner, finalize and package outputs.')
    args = parser.parse_args()
    config = load_config(args.config, args.output_dir)
    if args.background_supervisor:
        from scripts.supervise_stage_b import launch_supervisor
        print(json.dumps(launch_supervisor(config, args.stage_a_root.resolve(), args.sample_size), indent=2))
    else:
        report = run_stage_b(config, args.stage_a_root.resolve(), args.sample_size)
        print(json.dumps({k: report[k] for k in ('unique_urls', 'crawl', 'extraction', 'quality', 'next_stage')}, indent=2))
