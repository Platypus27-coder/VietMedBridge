"""Exercise shipped notebook subprocesses; no GPU performance claim."""
import ast
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vietmedbridge.artifacts import atomic_json, digest_json, read_json

REPO = Path(__file__).resolve().parents[1]
RUNTIME = read_json(REPO / "configs/kaggle_runtime.json")


@pytest.fixture
def helpers():
    notebook = read_json(REPO / "notebooks/04_kaggle_embedding_worker.ipynb")
    tree = ast.parse(next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"fresh_kaggle_python", "configure_kaggle_cuda", "launch_kaggle_fresh"}]
    assert len(functions) == 3
    from vietmedbridge.content_embeddings import checked
    namespace = {name: globals()[name] for name in (
        "os", "Path", "subprocess", "sys", "json", "signal", "importlib", "read_json")}
    namespace["checked"] = checked
    exec(compile(ast.Module(body=functions, type_ignores=[]), "kaggle-worker-notebook", "exec"), namespace)
    return namespace


def test_new_python_does_not_inherit_loaded_cpu_torch(helpers, monkeypatch):
    loaded_cpu = SimpleNamespace(__version__="2.11.0+cpu")
    monkeypatch.setitem(sys.modules, "torch", loaded_cpu)
    result = helpers["fresh_kaggle_python"](REPO,
        "import json, sys; print(json.dumps({'torch_loaded': 'torch' in sys.modules, 'args': json.loads(sys.argv[1])}))",
        {"path": "a path with spaces/Việt Nam", "account_id": 1}, capture_output=True)
    probe = json.loads(result.stdout)
    assert probe["torch_loaded"] is False
    assert probe["args"] == {"path": "a path with spaces/Việt Nam", "account_id": 1}
    assert sys.modules["torch"] is loaded_cpu


def test_captured_probe_failure_prints_original_error(helpers, capsys):
    with pytest.raises(subprocess.CalledProcessError) as error:
        helpers["fresh_kaggle_python"](REPO,
            "import sys; print('probe stdout'); print('actual CUDA failure', file=sys.stderr); sys.exit(7)",
            {}, capture_output=True)
    assert error.value.returncode == 7
    captured = capsys.readouterr()
    assert "probe stdout" in captured.out and "actual CUDA failure" in captured.err


@pytest.mark.parametrize("hung", [False, True])
def test_interrupt_stops_launcher_and_gpu_worker_process_group(helpers, hung):
    events = []
    class Process:
        pid = 123456
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
        def communicate(self):
            raise KeyboardInterrupt
        def wait(self, timeout):
            events.append(("wait", timeout))
            if hung and timeout == 30:
                raise subprocess.TimeoutExpired("launcher", timeout)
    helpers["subprocess"] = SimpleNamespace(Popen=Process, PIPE=subprocess.PIPE,
        TimeoutExpired=subprocess.TimeoutExpired)
    helpers["os"] = SimpleNamespace(environ={}, name="posix", pathsep=os.pathsep,
        killpg=lambda pid, sig: events.append((pid, sig)))
    helpers["signal"] = SimpleNamespace(SIGTERM=15, SIGKILL=9)
    with pytest.raises(KeyboardInterrupt):
        helpers["fresh_kaggle_python"](REPO, "pass", {})
    assert events[0] == (123456, 15)
    if hung:
        assert events[2:] == [(123456, 9), ("wait", 10)]
    else:
        assert events == [(123456, 15), ("wait", 30)]


def cuda_probe(**updates):
    return {"runtime": RUNTIME, "cuda_build": "12.6", "cuda_available": True,
            "gpu_count": 2, "devices": ["test-double-0", "test-double-1"]} | updates


def test_runtime_install_and_probe_continue_with_cpu_torch_loaded(helpers, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.11.0+cpu"))
    events = []
    command = [sys.executable, "-m", "pip", "install", "torch==" + RUNTIME["torch"]]
    helpers["runtime_install_commands"] = lambda runtime: [command]
    helpers["subprocess"] = SimpleNamespace(run=lambda args, **kwargs: events.append((args, kwargs)))
    def probe(checkout, source, arguments, **kwargs):
        assert events == [(command, {"check": True})]
        assert kwargs["capture_output"] is True
        # The real CUDA kernel operation must be in the subprocess, not the notebook.
        assert "torch.ones(1, device=" in source
        ast.parse(source)
        return SimpleNamespace(stdout=json.dumps(cuda_probe()))
    helpers["fresh_kaggle_python"] = probe
    assert helpers["configure_kaggle_cuda"](RUNTIME, 2, REPO) == cuda_probe()


@pytest.mark.parametrize("updates,message", [
    ({"cuda_available": False, "gpu_count": 0}, "đủ GPU"),
    ({"gpu_count": 1}, "đủ GPU"),
    ({"runtime": RUNTIME | {"torch": "2.11.0+cpu"}}, "runtime không khớp"),
])
def test_failed_gpu_probe_stops_before_embedding(helpers, updates, message):
    helpers["runtime_install_commands"] = lambda runtime: []
    helpers["fresh_kaggle_python"] = lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(cuda_probe(**updates)))
    with pytest.raises(RuntimeError, match=message):
        helpers["configure_kaggle_cuda"](RUNTIME, 2, REPO)


def test_launcher_runs_in_fresh_python_with_original_account_and_deadline(helpers, tmp_path, monkeypatch):
    """A fake child launcher records arguments; native GPU loop is tested elsewhere."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.11.0+cpu"))
    checkout = tmp_path / "checkout with spaces"
    package = checkout / "src/vietmedbridge"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(REPO / "src/vietmedbridge/artifacts.py", package / "artifacts.py")
    (package / "portable_embeddings.py").write_text(
        "import sys\nfrom pathlib import Path\nfrom .artifacts import atomic_json, read_json, digest_json\n"
        "def launch_kaggle(**args):\n"
        "    assert 'torch' not in sys.modules, 'inherited CPU kernel'\n"
        "    result = {'job': read_json(Path(args['job_dir']) / 'job.json')['manifest_sha256'], 'arguments': args}\n"
        "    result['manifest_sha256'] = digest_json(result)\n"
        "    atomic_json(Path(args['output']) / 'result-manifest.json', result)\n", encoding="utf-8")
    job_dir = tmp_path / "job"
    atomic_json(job_dir / "job.json", {"manifest_sha256": "a" * 64})
    result = helpers["launch_kaggle_fresh"](job_dir, checkout, output=tmp_path / "output",
        scratch=tmp_path / "scratch", session_started=123.5, hours=10.0,
        accounts=2, account_id=1, gpu_count=2)
    assert result["job"] == "a" * 64
    assert result["arguments"] == {"job_dir": str(job_dir), "checkout": str(checkout),
        "output": str(tmp_path / "output"), "scratch": str(tmp_path / "scratch"),
        "session_started": 123.5, "hours": 10.0, "accounts": 2, "account_id": 1, "gpu_count": 2}


def test_launcher_refuses_wrong_job_output(helpers, tmp_path):
    from vietmedbridge.content_embeddings import seal
    atomic_json(tmp_path / "job/job.json", {"manifest_sha256": "a" * 64})
    atomic_json(tmp_path / "output/result-manifest.json", seal({"job": "b" * 64}))
    helpers["fresh_kaggle_python"] = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="another job"):
        helpers["launch_kaggle_fresh"](tmp_path / "job", REPO, output=tmp_path / "output",
            scratch=tmp_path / "scratch", session_started=123.5, hours=10.0,
            accounts=2, account_id=1, gpu_count=2)
