"""A notebook resume must never fall through to corpus preparation or payload reads."""
import ast
from pathlib import Path

import nbformat
import pytest

from test_submission_workflow import prepared, frozen, external, setup_workflow
from scripts.submission_resume_only import restore_submission_preparation_only
from vietmedbridge import submission_workflow
from vietmedbridge.artifacts import atomic_json, read_json


def forbid_heavy_reads(monkeypatch):
    original = Path.open
    def open_metadata(path, *args, **kwargs):
        if (path.suffix in {'.parquet', '.npy', '.sqlite'} or path.name == 'catalog.json'
            or 'model_cache' in path.parts):
            pytest.fail(f'Resume tried to read payload: {path}')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_metadata)
    monkeypatch.setattr(submission_workflow, 'prepare_current_submission',
        lambda *a, **k: pytest.fail('Resume started CPU preparation'))


@pytest.mark.parametrize('missing_pointer', [False, True])
def test_restore_uses_metadata_only_after_fresh_runtime(prepared, monkeypatch, missing_pointer):
    p = prepared
    report = setup_workflow(p, monkeypatch)
    pointer = p.root / 'retrieval/current_submission.json'
    if missing_pointer:
        pointer.unlink()
    forbid_heavy_reads(monkeypatch)
    assert restore_submission_preparation_only(p.root, p.checkout) == report
    assert read_json(pointer) == report
    assert all(count == 0 for count in p.counters.values())


@pytest.mark.parametrize('fault', ['missing_reports', 'new_source', 'different_config'])
def test_mismatch_stops_without_touching_paid_payloads(prepared, monkeypatch, fault):
    p = prepared
    report = setup_workflow(p, monkeypatch)
    if fault == 'missing_reports':
        for path in (p.root / 'retrieval/current_submission.json',
                     p.root / 'processed' / report['build_run'] / 'submission_preparation.json',
                     p.root / 'retrieval/cpu_preparation' / (report['candidate_manifest_sha256'][:16] + '.json')):
            path.unlink()
    elif fault == 'new_source':
        atomic_json(p.root / 'active_data_batch.json', {'state': 'COMPLETE',
            'builds': [{'state': 'COMPLETE', 'build_run': 'another-completed-build'}]})
    else:
        path = p.checkout / 'configs/retrieval_full.json'
        config = read_json(path)
        atomic_json(path, config | {'configuration_changed': True})
    before = {path: path.stat().st_size for path in p.root.rglob('*') if path.is_file()}
    forbid_heavy_reads(monkeypatch)
    with pytest.raises(RuntimeError, match='RESUME_ONLY_STOP'):
        restore_submission_preparation_only(p.root, p.checkout)
    assert before == {path: path.stat().st_size for path in p.root.rglob('*') if path.is_file()}
    assert all(count == 0 for count in p.counters.values())


def test_notebook_embeds_the_guard_for_old_code_locked_checkouts():
    root = Path(__file__).resolve().parents[1]
    book = nbformat.read(root / 'notebooks/04_colab_submit_available_data.ipynb', as_version=4)
    source = [c.source for c in book.cells if c.cell_type == 'code'][1]
    ast.parse(source)
    assert (root / 'scripts/submission_resume_only.py').read_text(encoding='utf-8').strip() in source
    assert source.index('ALLOW_CPU_PREPARATION = False') < source.index('if ALLOW_CPU_PREPARATION:')
