"""Select a saved checkpoint explicitly; required restores fail before crawling."""
from pathlib import Path


def select_restore(input_root, shard_id, *, explicit=None, auto=True,
                   required=False, expected_sha256=None):
    selected = Path(explicit) if explicit else None
    if selected is None and auto:
        candidates = sorted(Path(input_root).rglob(f'vibiomir_shard_{shard_id:05d}.tar'))
        if len(candidates) > 1:
            raise ValueError('Multiple previous versions found; set RESTORE_BUNDLE explicitly.')
        selected = candidates[0] if candidates else None
    if selected is None:
        if required:
            raise FileNotFoundError('Resume requires the previous tar + sha256 mounted as input; refusing a fresh crawl.')
        return None
    if not selected.is_file():
        raise FileNotFoundError(selected)
    tokens = selected.with_suffix('.tar.sha256').read_text(encoding='utf-8').split()
    if not tokens or len(tokens[0]) != 64:
        raise ValueError('Saved checkpoint is missing a valid SHA256 sidecar.')
    if expected_sha256 is not None and tokens[0] != expected_sha256:
        raise ValueError('Mounted checkpoint does not match the selected previous run.')
    return str(selected)
