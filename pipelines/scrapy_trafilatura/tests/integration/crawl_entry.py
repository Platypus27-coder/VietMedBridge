import argparse
import json
from pathlib import Path

from src.crawler.entry import crawl_entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--origins', required=True)
    args = parser.parse_args()
    origins = frozenset(tuple(origin) for origin in json.loads(args.origins))
    if any(host != '127.0.0.1' for _, host, _ in origins):
        raise ValueError('Test harness only accepts its explicit IPv4 loopback origin.')
    crawl_entry(args.config, args.shard, args.stop_after, origins)
