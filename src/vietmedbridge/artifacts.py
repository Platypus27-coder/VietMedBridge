"""Atomic, hashed checkpoints; temporary work stays off Google Drive."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(local_path: str | Path, destination: str | Path) -> str:
    """Copy a completed local artifact, then atomically publish its filename."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid4().hex}.tmp")
    try:
        shutil.copyfile(local_path, temporary)
        checksum = sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return checksum


def verify_file(path: str | Path, checksum: str) -> None:
    if not Path(path).is_file() or sha256_file(path) != checksum:
        raise ValueError(f"Missing or changed artifact: {path}")


def local_workspace(work_dir: str | Path | None = None):
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="vmb-", dir=work_dir)


def runtime_versions() -> dict[str, str]:
    result = {}
    for name in ("vietmedbridge", "datasets", "huggingface-hub", "pyarrow", "duckdb",
                 "httpx", "trafilatura", "pypdf", "transformers", "tokenizers",
                 "lxml", "beautifulsoup4", "langdetect", "scrapling", "crawl4ai",
                 "playwright", "patchright", "playwright-stealth", "curl_cffi"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def code_fingerprint() -> str:
    package = Path(__file__).parent
    return digest_json({path.name: sha256_file(path) for path in sorted(package.glob("*.py"))})
