import pytest

from src.utils.restore_input import select_restore


def test_required_resume_refuses_missing_input(tmp_path):
    assert select_restore(tmp_path, 0) is None
    with pytest.raises(FileNotFoundError, match='refusing a fresh crawl'):
        select_restore(tmp_path, 0, required=True)


def test_restore_requires_sidecar_and_expected_checkpoint(tmp_path):
    archive = tmp_path/'vibiomir_shard_00000.tar'
    archive.write_bytes(b'fixture')
    with pytest.raises(FileNotFoundError):
        select_restore(tmp_path, 0, required=True)
    checksum = archive.with_suffix('.tar.sha256')
    checksum.write_text('a'*64+'  '+archive.name)
    assert select_restore(tmp_path, 0, required=True, expected_sha256='a'*64) == str(archive)
    with pytest.raises(ValueError, match='does not match'):
        select_restore(tmp_path, 0, required=True, expected_sha256='b'*64)


def test_multiple_outputs_require_explicit_selection(tmp_path):
    for name in ('v1', 'v2'):
        folder = tmp_path/name
        folder.mkdir()
        (folder/'vibiomir_shard_00000.tar').write_bytes(b'fixture')
        (folder/'vibiomir_shard_00000.tar.sha256').write_text('a'*64)
    with pytest.raises(ValueError, match='Multiple previous versions'):
        select_restore(tmp_path, 0, required=True)
    chosen = tmp_path/'v1/vibiomir_shard_00000.tar'
    assert select_restore(tmp_path, 0, explicit=chosen, required=True) == str(chosen)
