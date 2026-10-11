"""Preflight and cache behavior with a fake remote API; no Kaggle/GPU claims."""
from pathlib import Path
from contextlib import contextmanager
import json
import re
from types import SimpleNamespace

import pytest

from vietmedbridge.artifacts import digest_json
from vietmedbridge.kaggle_outputs import (
    MANIFEST_PATTERN, _download_pages, _stream_file, download_kaggle_outputs, inspect_kaggle_output, output_handles,
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


def test_version_handle_and_zero_receipts_reported_without_recounting(tmp_path, capsys, paged_api):
    handle = "owner/notebook/2"
    assert output_handles(f" {handle}, second/notebook ") == [handle, "second/notebook"]
    _, modern_api, _ = paged_api
    api = modern_api([files(result(parts=0))])
    probe = inspect_kaggle_output(api, handle, tmp_path)
    assert probe["manifest"]["receipts"] == []
    assert "saved_parts=0" in capsys.readouterr().out
    assert api.calls[-1][1] == "probe"


def test_manifest_filter_is_recursive_and_does_not_fetch_vectors():
    assert re.search(MANIFEST_PATTERN, "result-manifest.json")
    assert re.search(MANIFEST_PATTERN, "vmb_checkpoints/result-manifest.json")
    assert not re.search(MANIFEST_PATTERN, "data/vector.npy")


class HttpResponse:
    def __init__(self, content, *, length=None, fail=False):
        self.content = content
        self.headers = {"Content-Length": str(len(content) if length is None else length)}
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        if self.fail:
            raise RuntimeError("HTTP failure")

    def iter_content(self, chunk_size):
        # Test the streaming path, not response.content buffering.
        for start in range(0, len(self.content), 3):
            yield self.content[start:start + 3]


@pytest.fixture
def paged_api(monkeypatch):
    """Old public API cannot accept page_token; its SDK still can."""
    from vietmedbridge import kaggle_outputs
    import requests

    monkeypatch.setattr(kaggle_outputs, "_sdk_request", lambda: SimpleNamespace(version_label=None))
    fetched, content = [], {}

    def get(url, **kwargs):
        fetched.append(url)
        assert kwargs["stream"] is True
        return HttpResponse(content[url])

    monkeypatch.setattr(requests, "get", get)

    class BasePaged(FakeApi):
        def __init__(self, pages):
            self.pages = pages
            self.tokens, self.sdk_requests = [], []
            super().__init__({"owner/notebook": pages[0]})

        def _public_page(self, handle, path, token, options):
            self.tokens.append(token)
            index = int(token.split("-")[1]) if token else 0
            self.remote[handle] = self.pages[index]
            paths, _ = FakeApi.kernels_output(self, handle, path, **options)
            return paths, f"page-{index + 1}" if index + 1 < len(self.pages) else ""

        @contextmanager
        def build_kaggle_client(self):
            service = SimpleNamespace(list_kernel_session_output=self._sdk_page)
            yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=service))

        def _sdk_page(self, request):
            self.sdk_requests.append(request)
            assert (request.user_name, request.kernel_slug) == ("owner", "notebook")
            index = int(request.page_token.split("-")[1]) if request.page_token else 0
            entries = []
            for name, text in self.pages[index].items():
                url = f"memory://{index}/{name}"
                content[url] = text.encode()
                entries.append(SimpleNamespace(file_name=name, url=url))
            return SimpleNamespace(files=entries, next_page_token=(
                f"page-{index + 1}" if index + 1 < len(self.pages) else ""))

    class OldPaged(BasePaged):
        # Exact public signature of kaggle==2.0.2: no page_token parameter.
        def kernels_output(self, handle, path, *, file_pattern=None, force=False, quiet=True):
            return self._public_page(handle, path, None,
                {"file_pattern": file_pattern, "force": force, "quiet": quiet})

    class NewPaged(BasePaged):
        def kernels_output(self, handle, path, *, file_pattern=None, force=False, quiet=True, page_token=None):
            return self._public_page(handle, path, page_token,
                {"file_pattern": file_pattern, "force": force, "quiet": quiet})

    return OldPaged, NewPaged, fetched


@pytest.mark.parametrize("modern", [False, True])
def test_manifest_on_last_page_after_empty_filtered_pages(tmp_path, paged_api, modern):
    old, new, fetched = paged_api
    manifest = result()
    pages = [{"vmb_checkpoints/data/first.npy": "first vector"}, {},
             {"vmb_checkpoints/data/last.npy": "last vector"},
             {"vmb_checkpoints/result-manifest.json": json.dumps(manifest)}]
    api = (new if modern else old)(pages)
    probe = inspect_kaggle_output(api, "owner/notebook", tmp_path)
    assert probe["manifest"] == manifest
    if modern:
        assert api.tokens == ["", "page-1", "page-2", "page-3"]
        assert not api.sdk_requests
    else:
        assert [request.page_token for request in api.sdk_requests] == ["page-1", "page-2", "page-3"]
        assert len(fetched) == 1 and fetched[0].endswith("result-manifest.json")


@pytest.mark.parametrize("modern", [False, True])
def test_bulk_download_includes_late_vector_and_reuses_complete_files(tmp_path, paged_api, modern):
    old, new, _ = paged_api
    pages = [{"vmb_checkpoints/data/first.npy": "first vector"}, {},
             {"vmb_checkpoints/data/last.npy": "last vector"},
             {"vmb_checkpoints/result-manifest.json": json.dumps(result())}]
    api = (new if modern else old)(pages)
    output = download_kaggle_outputs(api, ["owner/notebook"], tmp_path)[0]
    vector = output / "data/last.npy"
    before = vector.stat().st_mtime_ns
    assert vector.read_text() == "last vector"
    assert (output / "data/first.npy").read_text() == "first vector"
    repeated = download_kaggle_outputs(api, ["owner/notebook"], tmp_path)[0]
    assert repeated == output and vector.stat().st_mtime_ns == before


def test_old_api_version_selection_uses_sdk_from_first_page(tmp_path, paged_api):
    old, _, _ = paged_api
    api = old([{"vmb_checkpoints/result-manifest.json": json.dumps(result())}])
    probe = inspect_kaggle_output(api, "owner/notebook/2", tmp_path)
    assert probe["manifest"] == result()
    assert not api.tokens and api.sdk_requests[0].version_label == "v2"


def test_repeated_token_stops_instead_of_looping_forever(tmp_path):
    class Repeated:
        def kernels_output(self, handle, path, *, page_token=None, **kwargs):
            return [], "repeated-token"
    with pytest.raises(RuntimeError, match="lặp token"):
        _download_pages(Repeated(), "owner/notebook", tmp_path)


def test_sdk_rejects_traversal_before_fetch_or_write(tmp_path, paged_api):
    old, _, fetched = paged_api
    api = old([{}, {"../result-manifest.json": json.dumps(result())}])
    with pytest.raises(RuntimeError):
        inspect_kaggle_output(api, "owner/notebook", tmp_path)
    assert not fetched and not (tmp_path / "result-manifest.json").exists()


@pytest.mark.parametrize("failure", ["http", "short"])
def test_failed_stream_preserves_previous_file_and_removes_temporary(tmp_path, monkeypatch, failure):
    import requests

    target = tmp_path / "vector.npy"
    target.write_bytes(b"complete previous vector")
    monkeypatch.setattr(requests, "get", lambda *_, **__: HttpResponse(
        b"new bytes", length=100 if failure == "short" else None, fail=failure == "http"))
    with pytest.raises(RuntimeError):
        _stream_file("https://example.invalid/signed-file", target, force=True)
    assert target.read_bytes() == b"complete previous vector"
    assert list(tmp_path.iterdir()) == [target]
