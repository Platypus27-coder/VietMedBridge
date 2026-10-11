"""Preflight and cache behavior with a fake remote API; no Kaggle/GPU claims."""
from pathlib import Path
import json
import re
from types import SimpleNamespace

import pytest

from vietmedbridge.artifacts import digest_json
from vietmedbridge.kaggle_outputs import (
    MANIFEST_PATTERN, download_kaggle_outputs, inspect_kaggle_output, output_handles,
)


def result(account_id=0, *, job="a" * 64, parts=2):
    value = {"schema": "portable-result-v1", "job": job,
             "assignment": {"account_id": account_id}, "state": "CHECKPOINTED_PARTIAL",
             "receipts": [{"path": f"receipts/bge-{i}.json"} for i in range(parts)]}
    return {**value, "manifest_sha256": digest_json(value)}


def files(manifest):
    return {"vmb_checkpoints/result-manifest.json": json.dumps(manifest),
            "vmb_checkpoints/data/vector.npy": "fake vector payload for download test only"}


class FakeApi:
    def __init__(self, remote, *, old_api=False):
        self.remote, self.calls, self.old_api = remote, [], old_api
        self.before_bulk = None

    def kernels_status(self, handle):
        self.calls.append((handle, "status"))
        return SimpleNamespace(status="complete")

    def kernels_output(self, handle, path, *, file_pattern=None, force=False, quiet=True):
        self.calls.append((handle, "probe" if file_pattern else "bulk", force))
        if file_pattern is None and self.before_bulk:
            self.before_bulk(handle)
        returned = []
        for name, text in self.remote[handle].items():
            if file_pattern and not re.search(file_pattern, name):
                continue
            destination = Path(path) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if force or not destination.exists():
                destination.write_text(text, encoding="utf-8")
            returned.append(str(destination))
        # Kaggle also returns the log even when file_pattern filters every data file.
        log = Path(path) / "notebook.log"
        log.write_text("fake remote log", encoding="utf-8")
        returned.append(str(log))
        return returned if self.old_api else (returned, "")


@pytest.mark.parametrize("old_api", [False, True])
def test_preflights_both_accounts_before_any_large_download(tmp_path, old_api):
    api = FakeApi({"owner-a/notebook-a": files(result(0)),
                   "owner-b/notebook-b": files(result(1))}, old_api=old_api)
    handles = list(api.remote)
    outputs = download_kaggle_outputs(api, handles, tmp_path)
    assert len(outputs) == 2
    assert [event[1] for event in api.calls] == ["status", "probe", "status", "probe", "bulk", "bulk"]
    assert all((path / "data/vector.npy").is_file() for path in outputs)
    assert all(event[2] is False for event in api.calls if event[1] == "bulk")


def test_missing_remote_manifest_names_account_and_never_downloads_vectors(tmp_path, capsys):
    api = FakeApi({"owner-a/notebook-a": {}, "owner-b/notebook-b": files(result(1))})
    with pytest.raises(RuntimeError, match="owner-a/notebook-a.*0 result-manifest"):
        download_kaggle_outputs(api, list(api.remote), tmp_path)
    assert not any(event[1] == "bulk" for event in api.calls)
    assert ("owner-b/notebook-b", "status") in api.calls
    log = capsys.readouterr().out
    assert "remote_manifests=0" in log and "status=complete" in log


def test_remote_ambiguity_reports_exact_paths_before_vectors(tmp_path):
    remote = files(result()) | {"other/result-manifest.json": json.dumps(result())}
    api = FakeApi({"owner/notebook": remote})
    with pytest.raises(RuntimeError, match="2 result-manifest") as error:
        download_kaggle_outputs(api, list(api.remote), tmp_path)
    assert "other/result-manifest.json" in str(error.value)
    assert not any(event[1] == "bulk" for event in api.calls)


def test_cached_manifests_cannot_hide_missing_remote_outputs(tmp_path):
    handle = "owner/notebook"
    cached = tmp_path / "outputs" / digest_json(handle)[:16] / "vmb_checkpoints"
    cached.mkdir(parents=True)
    (cached / "result-manifest.json").write_text(json.dumps(result()))
    with pytest.raises(RuntimeError, match="0 result-manifest"):
        download_kaggle_outputs(FakeApi({handle: {}}), [handle], tmp_path)


def test_stale_extra_local_manifest_does_not_create_false_ambiguity(tmp_path):
    handle, manifest = "owner/notebook", result()
    cached = tmp_path / "outputs" / digest_json(handle)[:16]
    for relative, text in files(manifest).items():
        destination = cached / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    stale = cached / "old/result-manifest.json"
    stale.parent.mkdir()
    stale.write_text("unrelated stale file")
    vector = cached / "vmb_checkpoints/data/vector.npy"
    before = vector.stat().st_mtime_ns
    outputs = download_kaggle_outputs(FakeApi({handle: files(manifest)}), [handle], tmp_path)
    assert outputs == [cached.resolve() / "vmb_checkpoints"]
    assert vector.stat().st_mtime_ns == before
    assert stale.read_text() == "unrelated stale file"


@pytest.mark.parametrize("different_job", [False, True])
def test_mismatched_accounts_or_job_fail_before_bulk(tmp_path, different_job):
    api = FakeApi({"owner-a/notebook-a": files(result(0)),
                   "owner-b/notebook-b": files(result(1 if different_job else 0,
                                                    job="b" * 64 if different_job else "a" * 64))})
    with pytest.raises(RuntimeError, match="job khác nhau|trùng KAGGLE_ACCOUNT_ID"):
        download_kaggle_outputs(api, list(api.remote), tmp_path)
    assert not any(event[1] == "bulk" for event in api.calls)


def test_latest_version_changed_after_preflight_is_rejected(tmp_path):
    handle = "owner/notebook"
    api = FakeApi({handle: files(result())})
    api.before_bulk = lambda _: api.remote[handle].update(files(result(parts=3)))
    with pytest.raises(RuntimeError, match="Outputs đổi sau preflight"):
        download_kaggle_outputs(api, [handle], tmp_path)


def test_new_manifest_does_not_overwrite_previous_version_cache(tmp_path):
    handle = "owner/notebook"
    api = FakeApi({handle: files(result())})
    first = download_kaggle_outputs(api, [handle], tmp_path)[0]
    before = (first / "result-manifest.json").read_bytes()
    api.remote[handle] = files(result(parts=3))
    second = download_kaggle_outputs(api, [handle], tmp_path)[0]
    assert first != second and (first / "result-manifest.json").read_bytes() == before


def test_manifest_integrity_checked_before_vector_download(tmp_path):
    broken = result()
    broken["job"] = "b" * 64
    api = FakeApi({"owner/notebook": files(broken)})
    with pytest.raises(RuntimeError, match="không hợp lệ"):
        download_kaggle_outputs(api, list(api.remote), tmp_path)
    assert not any(event[1] == "bulk" for event in api.calls)


def test_permission_failure_names_notebook_without_echoing_credentials(tmp_path, capsys):
    class Denied(FakeApi):
        def kernels_status(self, handle):
            raise ValueError("private credential detail must not be printed")
    with pytest.raises(RuntimeError, match="owner/notebook.*KAGGLE_JSON"):
        download_kaggle_outputs(Denied({}), ["owner/notebook"], tmp_path)
    assert "private credential" not in capsys.readouterr().out


@pytest.mark.parametrize("value", ["", "owner/notebook,owner/notebook", "https://kaggle.com/code/a/b",
                                   "owner/notebook?scriptVersionId=357105312", "owner/notebook/../x"])
def test_rejects_invalid_handles_before_api(value):
    with pytest.raises(ValueError):
        output_handles(value)


def test_version_handle_and_zero_receipts_reported_without_recounting(tmp_path, capsys):
    handle = "owner/notebook/2"
    assert output_handles(f" {handle}, second/notebook ") == [handle, "second/notebook"]
    api = FakeApi({handle: files(result(parts=0))})
    probe = inspect_kaggle_output(api, handle, tmp_path)
    assert probe["manifest"]["receipts"] == []
    assert "saved_parts=0" in capsys.readouterr().out
    assert api.calls[-1][1] == "probe"


def test_manifest_filter_is_recursive_and_does_not_fetch_vectors():
    assert re.search(MANIFEST_PATTERN, "result-manifest.json")
    assert re.search(MANIFEST_PATTERN, "vmb_checkpoints/result-manifest.json")
    assert not re.search(MANIFEST_PATTERN, "data/vector.npy")
