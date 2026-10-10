"""Exercise the generated bootstrap before network, mounts or package installs."""
import ast
import builtins
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/04_kaggle_embedding_worker.ipynb"


def bootstrap_platform(environ, shell_module, requested="auto"):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(source)
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CHECKOUT" for target in node.targets
        ):
            break
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUNTIME_PLATFORM" for target in node.targets
        ):
            node.value = ast.Constant(requested)
        prefix.append(node)

    class RuntimePath(PurePosixPath):
        def is_dir(self):
            # Reproduce an existing /kaggle/input on a Colab runtime.
            return str(self) == "/kaggle/input"

    shell = type("Shell", (), {"__module__": shell_module})()
    replacements = {
        "os": SimpleNamespace(environ=environ),
        "IPython": SimpleNamespace(get_ipython=lambda: shell),
        "pathlib": SimpleNamespace(Path=RuntimePath),
    }

    def runtime_import(name, *args, **kwargs):
        if name in replacements:
            return replacements[name]
        return builtins.__import__(name, *args, **kwargs)

    namespace = {"__builtins__": dict(vars(builtins), __import__=runtime_import)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])),
                 str(NOTEBOOK), "exec"), namespace)
    return namespace["IS_KAGGLE"]


@pytest.mark.parametrize("environ,shell_module,expected", [
    ({}, "google.colab._shell", False),
    ({"COLAB_RELEASE_TAG": "release-colab"}, "ipykernel.zmqshell", False),
    ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, "google.colab._shell", False),
    ({"COLAB_RELEASE_TAG": "release-colab", "KAGGLE_KERNEL_RUN_TYPE": "Batch"}, "ipykernel.zmqshell", False),
    ({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, "ipykernel.zmqshell", True),
    ({"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, "ipykernel.zmqshell", True),
])
def test_detects_active_runtime_even_with_kaggle_input_directory(environ, shell_module, expected):
    assert bootstrap_platform(environ, shell_module) is expected


@pytest.mark.parametrize("requested,expected", [("colab", False), ("kaggle", True)])
def test_explicit_platform_handles_custom_runtimes(requested, expected):
    assert bootstrap_platform({}, "custom.shell", requested) is expected


def test_directory_alone_does_not_classify_unknown_runtime_as_kaggle():
    with pytest.raises(RuntimeError, match="Không nhận diện được runtime"):
        bootstrap_platform({}, "custom.shell")


def test_invalid_platform_stops_before_network_or_mount():
    with pytest.raises(ValueError, match="RUNTIME_PLATFORM"):
        bootstrap_platform({}, "custom.shell", "typo")
