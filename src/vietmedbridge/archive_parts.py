"""Restore checksum-bound archive parts on local Colab disk."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .artifacts import read_json, sha256_file


def parts_manifest(directory):
    root = Path(directory)
    value = read_json(root / "archive_manifest.json")
    name = value.get("filename", "")
    if (value.get("schema") != "vietmedbridge-archive-parts-v1"
        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.tar", name)
        or not re.fullmatch(r"[a-f0-9]{64}", value.get("sha256", ""))
        or not value.get("parts") or len(value["parts"]) > 10000):
        raise ValueError("Invalid archive parts manifest.")
    for i, part in enumerate(value["parts"]):
        if (part.get("name") != f"{name}.part{i:05d}"
            or type(part.get("bytes")) is not int or part["bytes"] <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", part.get("sha256", ""))):
            raise ValueError("Invalid archive part name/size/checksum.")
        path = root / part["name"]
        if not path.is_file() or path.stat().st_size != part["bytes"]:
            raise ValueError(f"Missing or incomplete archive part: {path}")
    if sum(p["bytes"] for p in value["parts"]) != value.get("bytes"):
        raise ValueError("Archive part sizes do not sum to its original size.")
    return value


def restore_archive(directory, cache_dir):
    """Never concatenate in RAM or write a second full archive back to Drive."""
    root = Path(directory)
    manifest = parts_manifest(root)
    target = Path(cache_dir) / manifest["sha256"][:16] / manifest["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == manifest["bytes"] and sha256_file(target) == manifest["sha256"]:
        return target
    temporary = target.with_suffix(".tar.incomplete")
    archive_hash = hashlib.sha256()
    written = 0
    with temporary.open("wb") as outgoing:
        for part in manifest["parts"]:
            part_hash = hashlib.sha256()
            count = 0
            with (root / part["name"]).open("rb") as incoming:
                while data := incoming.read(1024**2):
                    outgoing.write(data)
                    part_hash.update(data)
                    archive_hash.update(data)
                    count += len(data)
            if count != part["bytes"] or part_hash.hexdigest() != part["sha256"]:
                raise ValueError(f"Archive part checksum mismatch: {part['name']}")
            written += count
            print(f"Archive restore: {written:,}/{manifest['bytes']:,} bytes", flush=True)
    if written != manifest["bytes"] or archive_hash.hexdigest() != manifest["sha256"]:
        raise ValueError("Restored archive checksum mismatch.")
    temporary.replace(target)
    return target
