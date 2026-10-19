"""Use real local files to prove mountpoint repair never deletes their contents."""
from pathlib import Path

import nbformat
import pytest

from scripts.colab_drive_mount import preserve_unmounted_drive_directory, safe_colab_mount, safe_colab_remount


class Drive:
    def __init__(self, mount, *, second_error=None):
        self.path, self.calls, self.flushed = mount, [], False
        self.second_error = second_error

    def flush_and_unmount(self):
        self.flushed = True

    def mount(self, path, **kwargs):
        self.calls.append((path, kwargs))
        assert Path(path) == self.path
        if len(self.calls) > 1 and self.second_error:
            raise self.second_error
        if self.path.exists() and next(self.path.iterdir(), None) is not None:
            raise ValueError('Mountpoint must not already contain files')


@pytest.fixture
def local(tmp_path_factory):
    root = tmp_path_factory.mktemp('mt')
    mount = root / 'drive'
    mount.mkdir()
    nested = mount / 'MyDrive/VietMedBridge/data'
    nested.mkdir(parents=True)
    (nested / 'unpublished-local-block.npy').write_bytes(b'preserve every byte')
    table = root / 'mountinfo'
    table.write_text('1 0 8:0 / / rw - ext4 /dev/root rw\n')
    return root, mount, nested, table


def test_nonempty_unmounted_directory_is_preserved_and_exact_mountpoint_retried(local):
    root, mount, nested, table = local
    drive = Drive(mount)
    backup = safe_colab_remount(mount, drive_api=drive, content_root=root, mountinfo_file=table)
    assert drive.flushed and len(drive.calls) == 2
    assert backup.parent == root and backup.name.startswith('vmb_drive_unmounted_')
    assert (backup / nested.relative_to(mount) / 'unpublished-local-block.npy').read_bytes() == b'preserve every byte'
    assert list(mount.iterdir()) == []
    assert all(call[0] == str(mount) for call in drive.calls)


@pytest.mark.parametrize('nested_mount', [False, True])
def test_active_or_nested_kernel_mount_prevents_any_directory_move(local, nested_mount):
    root, mount, nested, table = local
    point = nested if nested_mount else mount
    table.write_text(f'1 0 8:0 / / rw - ext4 /dev/root rw\n2 1 0:1 / {point.as_posix()} rw - fuse.drive drive rw\n')
    with pytest.raises(RuntimeError, match='Kernel mount still active'):
        safe_colab_remount(mount, drive_api=Drive(mount), content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').read_bytes() == b'preserve every byte'
    assert not list(root.glob('vmb_drive_unmounted_*'))


def test_mount_table_unavailable_refuses_to_move_local_files(local):
    root, mount, nested, table = local
    table.unlink()
    with pytest.raises(RuntimeError, match='Cannot verify kernel mount state'):
        safe_colab_mount(mount, drive_api=Drive(mount), content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').exists()


def test_malformed_mount_table_refuses_to_move(local):
    root, mount, nested, table = local
    table.write_text('invalid mount metadata\n')
    with pytest.raises(RuntimeError, match='Invalid kernel mount information'):
        safe_colab_mount(mount, drive_api=Drive(mount), content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').exists()


def test_empty_mount_table_does_not_count_as_unmounted(local):
    root, mount, nested, table = local
    table.write_text('')
    with pytest.raises(RuntimeError, match='Empty kernel mount information'):
        safe_colab_mount(mount, drive_api=Drive(mount), content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').exists()


def test_escaped_nested_mount_path_is_detected(local):
    root, mount, nested, table = local
    point = mount / 'nested mount'
    encoded = point.as_posix().replace(' ', r'\040')
    table.write_text(f'1 0 8:0 / / rw - ext4 /dev/root rw\n2 1 0:1 / {encoded} rw - fuse.drive drive rw\n')
    with pytest.raises(RuntimeError, match='Kernel mount still active'):
        safe_colab_mount(mount, drive_api=Drive(mount), content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').exists()


def test_credentials_error_is_not_treated_as_mountpoint_error(local):
    root, mount, nested, table = local
    class Credentials(Drive):
        def mount(self, *a, **k):
            raise ValueError('credential propagation was unsuccessful')
    with pytest.raises(ValueError, match='credential propagation'):
        safe_colab_mount(mount, drive_api=Credentials(mount), content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').exists()
    assert not list(root.glob('vmb_drive_unmounted_*'))


def test_auth_failure_after_preservation_still_keeps_backup(local):
    root, mount, _, table = local
    drive = Drive(mount, second_error=ValueError('authorization required'))
    with pytest.raises(ValueError, match='authorization required'):
        safe_colab_mount(mount, drive_api=drive, content_root=root, mountinfo_file=table)
    backups = list(root.glob('vmb_drive_unmounted_*'))
    assert len(backups) == 1
    assert (backups[0] / 'MyDrive/VietMedBridge/data/unpublished-local-block.npy').read_bytes() == b'preserve every byte'
    assert list(mount.iterdir()) == []


def test_empty_directory_mount_is_idempotent_and_needs_no_preservation(tmp_path):
    mount = tmp_path / 'drive'
    mount.mkdir()
    drive = Drive(mount)
    assert safe_colab_mount(mount, drive_api=drive) is None
    assert safe_colab_mount(mount, drive_api=drive) is None
    assert not list(tmp_path.glob('vmb_drive_unmounted_*'))


def test_move_is_restricted_to_exact_content_root_drive(local):
    root, mount, nested, table = local
    with pytest.raises(ValueError, match='only the named'):
        preserve_unmounted_drive_directory(nested, content_root=root, mountinfo_file=table)
    assert (nested / 'unpublished-local-block.npy').exists()


def test_bootstrap_and_gpu_cells_embed_identical_pure_mount_helper():
    root = Path(__file__).resolve().parents[1]
    helper = (root / 'scripts/colab_drive_mount.py').read_text(encoding='utf-8').strip()
    book = nbformat.read(root / 'notebooks/04_colab_submit_available_data.ipynb', as_version=4)
    codes = [c.source for c in book.cells if c.cell_type == 'code']
    for index in (0, 2, 3):
        assert helper in codes[index]
    assert codes[0].index('def safe_colab_mount') < codes[0].index('import vietmedbridge')
