from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def snapshot_binary_reader(path: Path):
    """Close the filesystem handle before parsing a live status snapshot."""
    if os.name != 'nt':
        with path.open('rb') as handle:
            return io.BytesIO(handle.read())
    import ctypes
    from ctypes import wintypes as w
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    kernel.CreateFileW.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        snapshot = stream.read()
    return io.BytesIO(snapshot)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows a brief reader/antivirus handle can deny replacement.
        # Retry only sharing/access errors; persistent failures still surface.
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                    raise
                time.sleep(min(0.025 * 2**attempt, 0.2))
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def write_parquet(path: Path, rows, schema: pa.Schema | None = None, batch_size: int = 10000) -> int:
    """Stream rows into one atomic Parquet file. Empty outputs retain their schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    count = 0
    writer = None
    try:
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                schema = table.schema
                writer = writer or pq.ParquetWriter(temporary, schema, compression='zstd')
                writer.write_table(table)
                count += len(batch)
                batch = []
        if batch or writer is None:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer = writer or pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
            count += len(batch)
        writer.close()
        writer = None
        with temporary.open('r+b') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return count
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def parquet_rows(path: Path, batch_size: int = 10000):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()
