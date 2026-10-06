"""Streaming, checksummed raw + manifest + checkpoint bundles for notebook sessions."""
import hashlib
import shutil
import tarfile
from pathlib import Path

from src.storage.io import atomic_bytes, sha256_file


def verify_bundle(bundle: Path, progress=None):
    count = size = 0
    with tarfile.open(bundle, 'r|') as archive:
        for member in archive:
            if not member.isfile() or not member.pax_headers.get('vibiomir.sha256'):
                raise ValueError('Bundle contains an unchecked/non-file member.')
            digest = hashlib.sha256()
            stream = archive.extractfile(member)
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
                size += len(chunk)
            if digest.hexdigest() != member.pax_headers['vibiomir.sha256']:
                raise ValueError(f'Bundle checksum mismatch: {member.name}')
            count += 1
            if progress:
                progress('bundle_verify', verified_files=count, verified_bytes=size)
            archive.members.clear()
    return {'files': count, 'payload_bytes': size}


def make_bundle(root: Path, destination: Path, progress=None):
    root, destination = root.resolve(), destination.resolve()
    if destination.is_relative_to(root):
        raise ValueError('Bundle must be outside its source directory.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tar.tmp')
    files = payload = 0
    try:
        with tarfile.open(temporary, 'w|', format=tarfile.PAX_FORMAT) as archive:
            for path in root.rglob('*'):
                relative = path.relative_to(root)
                if relative.parts[0] == 'crawl_jobs' or path.name in ('crawl_index.sqlite', 'extraction_index.sqlite'):
                    continue  # Reconstruct queues/indexes from durable events after restore.
                if path.is_symlink():
                    raise ValueError('Session bundle cannot contain symlinks.')
                if not path.is_file() or path.name == 'kaggle_writer.lock':
                    continue
                member = archive.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                member.pax_headers['vibiomir.sha256'] = sha256_file(path)
                with path.open('rb') as stream:
                    archive.addfile(member, stream)
                files += 1
                payload += member.size
                if progress:
                    progress('bundling', packed_files=files, payload_bytes=payload)
                archive.members.clear()  # Bound metadata memory as well as byte I/O.
        verification = verify_bundle(temporary, progress=progress)
        if verification != {'files': files, 'payload_bytes': payload}:
            raise ValueError('Bundle inventory mismatch.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if progress:
        progress('bundle_checksum', archive_bytes=destination.stat().st_size)
    checksum = sha256_file(destination)
    atomic_bytes(destination.with_suffix('.tar.sha256'), f'{checksum}  {destination.name}\n'.encode())
    return {**verification, 'archive': str(destination), 'sha256': checksum, 'verified': True}


def restore_bundle(bundle: Path, root: Path):
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Restore into an empty session directory.')
    checksum_file = bundle.with_suffix('.tar.sha256')
    expected = checksum_file.read_text(encoding='utf-8').split()[0]
    if sha256_file(bundle) != expected:
        raise ValueError('Session bundle SHA256 mismatch.')
    # Validate every member before making a resumable state visible.
    verify_bundle(bundle)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, 'r|') as archive:
        for member in archive:
            target = (root/member.name).resolve()
            if not target.is_relative_to(root) or not member.isfile():
                raise ValueError(f'Unsafe session bundle member: {member.name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, target.open('wb') as output:
                shutil.copyfileobj(stream, output, length=1024*1024)
            archive.members.clear()
