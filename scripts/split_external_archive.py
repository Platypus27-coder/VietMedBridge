"""Split an archive below both runtime (512 MiB) and connector (100 MiB) limits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_archive(source: Path, destination: Path, part_size=96 * 1024**2):
    if type(part_size) is not int or part_size < 1:
        raise ValueError("Part size must be a positive integer.")
    destination.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    archive_hash = hashlib.sha256()
    parts = []
    with source.open("rb") as incoming:
        while incoming.tell() < before.st_size:
            name = f"{source.name}.part{len(parts):05d}"
            final = destination / name
            temporary = final.with_suffix(final.suffix + ".tmp")
            part_hash = hashlib.sha256()
            written = 0
            with temporary.open("wb") as outgoing:
                while written < part_size:
                    data = incoming.read(min(1024**2, part_size - written))
                    if not data:
                        break
                    outgoing.write(data)
                    part_hash.update(data)
                    archive_hash.update(data)
                    written += len(data)
            temporary.replace(final)
            parts.append({"name": name, "bytes": written, "sha256": part_hash.hexdigest()})
            print(f"Prepared {name}: {written:,} bytes", flush=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source archive changed while splitting.")
    manifest = {"schema": "vietmedbridge-archive-parts-v1", "filename": source.name,
                "bytes": before.st_size, "sha256": archive_hash.hexdigest(),
                "part_size": part_size, "parts": parts}
    path = destination / "archive_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(split_archive(args.source, args.destination), indent=2))
