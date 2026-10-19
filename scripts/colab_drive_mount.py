"""Repair a nonempty, unmounted Colab Drive directory without deleting files.

Pure standard library so notebook setup can embed this before pip/package imports.
Kernel mount state must positively exclude this directory and any nested mounts
before an atomic rename preserves the complete local directory under /content.
"""
import os
from pathlib import Path
import re
from uuid import uuid4


def preserve_unmounted_drive_directory(mount, *, content_root='/content',
                                       mountinfo_file='/proc/self/mountinfo'):
    target, parent = Path(mount).absolute(), Path(content_root).resolve()
    if target.name != 'drive' or target.parent.resolve() != parent:
        raise ValueError('Drive recovery may preserve only the named content_root/drive directory.')
    if target.is_symlink():
        raise RuntimeError('Drive mountpoint is a symlink; refusing to move it.')
    try:
        entries = Path(mountinfo_file).read_text(encoding='utf-8').splitlines()
    except OSError as error:
        raise RuntimeError('Cannot verify kernel mount state; no Drive directory was moved.') from error
    if not entries:
        raise RuntimeError('Empty kernel mount information; no Drive directory was moved.')
    for line in entries:
        fields = line.split()
        if len(fields) < 10 or '-' not in fields[6:]:
            raise RuntimeError('Invalid kernel mount information; no Drive directory was moved.')
        name = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4])
        point = Path(name).absolute()
        if point == target or point.is_relative_to(target):
            raise RuntimeError(f'Kernel mount still active at {point}; refusing to move Drive data.')
    if os.path.ismount(target):
        raise RuntimeError('Drive is still a mountpoint; refusing to move it.')
    if not target.exists():
        return None
    if not target.is_dir():
        raise RuntimeError('Drive mountpoint is a file; refusing to move it.')
    if next(target.iterdir(), None) is None:
        return None
    backup = parent / ('vmb_drive_unmounted_' + uuid4().hex)
    if backup.exists() or backup.is_symlink():
        raise RuntimeError('Preservation path already exists; no Drive directory was moved.')
    # Same-parent rename retains every nested file and never follows child links.
    # Only an unmounted local directory is moved; original remote data is untouched.
    target.rename(backup)
    target.mkdir()
    print('Local unmounted Drive directory preserved at:', backup, flush=True)
    return backup


def safe_colab_mount(mount='/content/drive', *, force_remount=False, readonly=False,
                     drive_api=None, content_root='/content', mountinfo_file='/proc/self/mountinfo'):
    if drive_api is None:
        from google.colab import drive as drive_api
    try:
        drive_api.mount(str(mount), force_remount=force_remount, readonly=readonly)
        return None
    except ValueError as error:
        if 'Mountpoint must not already contain files' not in str(error):
            raise
    backup = preserve_unmounted_drive_directory(mount, content_root=content_root,
        mountinfo_file=mountinfo_file)
    drive_api.mount(str(mount), force_remount=True, readonly=readonly)
    return backup


def safe_colab_remount(mount, *, readonly=False, drive_api=None,
                       content_root='/content', mountinfo_file='/proc/self/mountinfo'):
    if drive_api is None:
        from google.colab import drive as drive_api
    try:
        drive_api.flush_and_unmount()
    except Exception as error:
        print('Drive flush/unmount did not complete:', type(error).__name__, flush=True)
    return safe_colab_mount(mount, force_remount=True, readonly=readonly, drive_api=drive_api,
        content_root=content_root, mountinfo_file=mountinfo_file)
