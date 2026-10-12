"""Drive disconnects must reconnect once and retain paid-for embedding parts."""
import ast
import errno
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vietmedbridge import colab_storage as storage, content_embeddings as embeddings
from vietmedbridge.artifacts import atomic_json, read_json, sha256_file


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    root, work = tmp_path / "drive/data", tmp_path / "work"
    atomic_json(root / "active_data_candidate.json", {"candidate_manifest_sha256":"original"})
    monkeypatch.setattr(storage, "_drive_mountpoint", lambda path:root)
    remounts = []
    monkeypatch.setattr(storage, "_remount", lambda mount, **kwargs:remounts.append((mount, kwargs)))
    return root, work, remounts


def disconnected(root):
    return OSError(107, "Transport endpoint is not connected", str(root / "checkpoint.json"))


def test_resume_after_disconnect_does_not_reencode_published_content(mounted, monkeypatch):
    root, work, remounts = mounted
    input_dir, output = root / "inputs", root / "vectors"
    input_dir.mkdir()
    parts = []
    for start, text in enumerate(("alpha", "beta", "gamma")):
        path = input_dir / f"units-{start}.parquet"
        pq.write_table(pa.Table.from_pylist([{"text":text}]), path)
        parts.append({"start":start, "rows":1, "path":path.name, "sha256":sha256_file(path)})
    inputs = embeddings.seal({"state":"COMPLETE", "input_count":3, "parts":parts})
    encoded = []
    class Encoder:
        identity = {"model_id":"test", "revision":"fixed"}
        dimension = 2
        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            return np.tile(np.array([[1., 0.]], np.float32), (len(texts), 1))
    failed = False
    def publish(path, value):
        nonlocal failed
        # Two vector blocks and the first complete map are already published.
        if Path(path).name == "map-000000000001.done.json" and not failed:
            failed = True
            raise disconnected(root)
        atomic_json(path, value)
    monkeypatch.setattr(embeddings, "atomic_json", publish)
    attempts = []
    def run():
        attempts.append(True)
        return embeddings.content_embeddings(root, input_dir, inputs, Encoder(), output, work_dir=work)
    result = storage.run_with_drive_recovery(run, root, work)
    assert result["state"] == "COMPLETE" and len(result["parts"]) == 3
    assert encoded == ["alpha", "beta", "gamma"]
    assert len(attempts) == 2 and len(remounts) == 1
    assert not list(work.glob("vmb-*"))  # both interrupted and resumed lookups closed
    report = read_json(next((work / "drive_recovery").glob("*.json")))
    assert report["state"] == "RECOVERED"


def test_preflight_repairs_mount_before_starting_model_stage(mounted, monkeypatch):
    root, work, remounts = mounted
    binding = storage._binding
    probes, model_loads = [], []
    def first_probe_fails(path):
        probes.append(True)
        if len(probes) == 1:
            raise disconnected(root)
        return binding(path)
    monkeypatch.setattr(storage, "_binding", first_probe_fails)
    result = storage.run_with_drive_recovery(lambda:model_loads.append(True) or "ready", root, work)
    assert result == "ready" and len(model_loads) == 1 and len(remounts) == 1


def test_remount_cannot_switch_the_active_corpus(mounted, monkeypatch):
    root, work, _ = mounted
    attempts = []
    def run():
        attempts.append(True)
        raise disconnected(root)
    monkeypatch.setattr(storage, "_remount", lambda *a, **k:
        atomic_json(root / "active_data_candidate.json", {"candidate_manifest_sha256":"another"}))
    with pytest.raises(ValueError, match="candidate changed"):
        storage.run_with_drive_recovery(run, root, work)
    assert len(attempts) == 1


def test_persistent_disconnect_stops_after_one_remount(mounted):
    root, work, remounts = mounted
    attempts = []
    def run():
        attempts.append(True)
        raise disconnected(root)
    with pytest.raises(storage.DriveUnavailableError, match="Giữ runtime"):
        storage.run_with_drive_recovery(run, root, work)
    assert len(attempts) == 2 and len(remounts) == 1
    assert read_json(next((work / "drive_recovery").glob("*.json")))["state"] == "DRIVE_UNAVAILABLE"


@pytest.mark.parametrize("error", [ValueError("corrupt checkpoint"), OSError(errno.ENOSPC, "disk full")])
def test_corruption_and_full_disk_are_not_treated_as_disconnect(mounted, error):
    root, work, remounts = mounted
    def run():
        raise error
    with pytest.raises(type(error)) as caught:
        storage.run_with_drive_recovery(run, root, work)
    assert caught.value is error and remounts == []


def test_unrelated_mount_failure_is_not_remounted(mounted):
    root, work, remounts = mounted
    error = OSError(107, "Transport endpoint is not connected", str(work / "other-mount/file"))
    def run():
        raise error
    with pytest.raises(OSError) as caught:
        storage.run_with_drive_recovery(run, root, work)
    assert caught.value is error and remounts == []


def test_decorated_stage_keeps_its_arguments_and_readonly_mode(mounted):
    root, work, remounts = mounted
    calls = []
    @storage.drive_recoverable(root_arg="data_root", work_arg="work_dir")
    def stage(data_root, *, work_dir, execution_mode="full"):
        calls.append((data_root, work_dir, execution_mode))
        if len(calls) == 1:
            raise disconnected(root)
        return "done"
    assert stage(root, work_dir=work, execution_mode="cached_strong") == "done"
    assert calls == [(root, work, "cached_strong")] * 2
    assert remounts == [(root, {"readonly":True})]


@pytest.fixture
def notebook_mount(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[1]
    notebook=read_json(repo/"notebooks/04_colab_retrieval_baseline.ipynb")
    bootstrap=next(c["source"] for c in notebook["cells"] if c["cell_type"]=="code")
    function=next(node for node in ast.parse(bootstrap).body if isinstance(node,ast.FunctionDef)
        and node.name=="_vmb_mount_data_root")
    calls=[]
    colab=ModuleType("google.colab")
    colab.drive=SimpleNamespace(mount=lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setitem(sys.modules,"google.colab",colab)
    context={"Path":Path,"json":json,"DRIVE_DATA_ROOT":str(tmp_path/"data")}
    exec(compile(ast.Module(body=[function],type_ignores=[]),"04-bootstrap-mount","exec"),context)
    root=Path(context["DRIVE_DATA_ROOT"])
    atomic_json(root/"active_data_candidate.json",{"build_run":"existing","candidate_manifest_sha256":"original"})
    return context["_vmb_mount_data_root"],root,calls


def test_uploaded_notebook_recovers_disconnected_drive_during_bootstrap(notebook_mount,monkeypatch):
    mount,root,calls=notebook_mount
    original=Path.read_bytes
    probes=[]
    def probe(path):
        probes.append(path)
        if len(probes)==1:
            raise disconnected(root)
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",probe)
    assert mount()==root
    assert len(probes)==2 and calls==[
        (("/content/drive",),{"force_remount":False,"readonly":False}),
        (("/content/drive",),{"force_remount":True,"readonly":False})]


def test_uploaded_notebook_stops_if_drive_stays_disconnected(notebook_mount,monkeypatch):
    mount,root,calls=notebook_mount
    def probe(path):
        raise disconnected(root)
    monkeypatch.setattr(Path,"read_bytes",probe)
    with pytest.raises(RuntimeError,match="Drive vẫn mất kết nối"):
        mount()
    assert len(calls)==2


def test_uploaded_notebook_does_not_create_empty_data_on_wrong_account(notebook_mount):
    mount,root,calls=notebook_mount
    (root/"active_data_candidate.json").unlink()
    root.rmdir()
    with pytest.raises(RuntimeError,match="Không thấy active candidate"):
        mount()
    assert not root.exists() and len(calls)==1


def test_uploaded_notebook_keeps_readonly_mode_when_remounting(notebook_mount,monkeypatch):
    mount,root,calls=notebook_mount
    original=Path.read_bytes
    attempts=[]
    def probe(path):
        attempts.append(True)
        if len(attempts)==1:
            raise disconnected(root)
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",probe)
    assert mount(readonly=True)==root
    assert all(call[1]["readonly"] for call in calls)
