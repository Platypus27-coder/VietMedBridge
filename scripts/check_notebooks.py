"""Validate notebook JSON, empty outputs and Python syntax, including top-level await."""

import ast
from pathlib import Path

import nbformat

for path in sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(f"Clear outputs before committing: {path.name}:{index}")
            compile(cell.source, f"{path.name}:cell-{index}", "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    print(f"OK {path.name}: {len(notebook.cells)} cells")
