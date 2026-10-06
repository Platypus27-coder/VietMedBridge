"""Build a standard Kaggle/Jupyter notebook from the maintained cell script."""
import json
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'notebooks/01_kaggle_stage_a.py')
    source = parser.parse_args().source
    cells, lines = [], []
    for line in source.read_text(encoding='utf-8').splitlines(keepends=True):
        if line.startswith('# %%') and lines:
            cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lines})
            lines = []
        lines.append(line)
    if lines:
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': lines})
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 4}
    destination = source.with_suffix('.ipynb')
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Built {destination.name}: {len(cells)} cells')
