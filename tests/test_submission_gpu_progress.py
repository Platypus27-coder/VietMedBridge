"""Parallel startup preserves the native manifest contract and payload checks."""
from pathlib import Path
import ast

import nbformat
import pytest
import scripts.submission_gpu_progress as adapter

from test_submission_workflow import prepared, frozen, external, setup_workflow
from test_submission_workflow import test_legacy_baseline_zip_and_resume as exercise_baseline
from scripts.submission_gpu_progress import read_bge_maps_parallel, submission_gpu_progress
from vietmedbridge import scale_baseline, submission_workflow
from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.content_embeddings import seal
from vietmedbridge.scale_benchmark import load_handoff


def context(p):
    _, candidate, inputs = load_handoff(p.root)
    config = read_json(p.checkout / 'configs/retrieval_full.json')
    return candidate, inputs, config


def test_parallel_map_result_is_identical_to_native_reader(prepared, monkeypatch):
    p = prepared
    setup_workflow(p, monkeypatch)
    candidate, inputs, config = context(p)
    expected = scale_baseline.imported_bge_manifest(p.root, candidate, inputs, config)
    messages = []
    assert read_bge_maps_parallel(p.root, candidate, inputs, config, progress=messages.append) == expected
    assert messages[0].startswith('BGE mapping check: 0/')
    assert messages[-1].endswith(f"valid maps {len(expected[1]['parts']):,}")


@pytest.mark.parametrize('fault', ['missing', 'payload', 'binding', 'identity'])
def test_parallel_maps_do_not_promote_partial_or_changed_vectors(prepared, monkeypatch, fault):
    p = prepared
    setup_workflow(p, monkeypatch)
    candidate, inputs, config = context(p)
    view, native = scale_baseline.imported_bge_manifest(p.root, candidate, inputs, config)
    assert native['parts']
    part = native['parts'][0]
    marker = view / f"map-{part['start']:012d}.done.json"
    if fault == 'missing':
        marker.unlink()
        _, actual = read_bge_maps_parallel(p.root, candidate, inputs, config)
        assert actual['state'] == 'IN_PROGRESS'
        assert len(actual['parts']) == len(native['parts']) - 1
    else:
        if fault == 'payload':
            with (view / part['path']).open('ab') as stream:
                stream.write(b'changed')
        elif fault == 'binding':
            saved = read_json(marker)
            saved['input_sha256'] = 'another-input'
            atomic_json(marker, seal({k: v for k, v in saved.items() if k != 'manifest_sha256'}))
        else:
            path = view / 'config.json'
            saved = read_json(path)
            saved['dimension'] = 1
            atomic_json(path, saved)
        with pytest.raises(ValueError):
            read_bge_maps_parallel(p.root, candidate, inputs, config)


def test_progress_restores_native_bindings_after_interrupt(prepared, monkeypatch, capsys):
    p = prepared
    original = scale_baseline.imported_bge_manifest
    native_context = submission_workflow._submission_context
    with pytest.raises(KeyboardInterrupt):
        with submission_gpu_progress(p.root, p.work):
            assert scale_baseline.imported_bge_manifest is not original
            raise KeyboardInterrupt()
    assert scale_baseline.imported_bge_manifest is original
    assert submission_workflow._submission_context is native_context
    assert read_json(p.root / 'retrieval/submission_progress.json')['state'] == 'STOPPED'
    assert 'SUBMISSION startup' in capsys.readouterr().out


def test_stalled_map_readers_do_not_block_keyboard_interrupt(prepared, monkeypatch):
    p = prepared
    setup_workflow(p, monkeypatch)
    candidate, inputs, config = context(p)
    shutdowns = []

    class StalledPool:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            return object()

        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(adapter, 'ThreadPoolExecutor', StalledPool)
    monkeypatch.setattr(adapter, 'as_completed', interrupt)
    with pytest.raises(KeyboardInterrupt):
        read_bge_maps_parallel(p.root, candidate, inputs, config)
    assert shutdowns == [{'wait': False, 'cancel_futures': True}]


def test_real_baseline_zip_works_under_the_scoped_adapter(prepared, monkeypatch):
    p = prepared
    original = scale_baseline.imported_bge_manifest
    with submission_gpu_progress(p.root, p.work):
        # Reuse the established CPU neural doubles and full ZIP/resume assertions.
        # Production CUDA guards remain intact.
        exercise_baseline(p, monkeypatch)
    assert read_json(p.root / 'retrieval/baseline_submission.json')['state'] == 'READY_FOR_MANUAL_UPLOAD'
    assert scale_baseline.imported_bge_manifest is original
    status = read_json(p.root / 'retrieval/submission_progress.json')
    assert status['state'] == 'CALL_FINISHED'
    assert status['candidate_manifest_sha256'] == context(p)[0]['candidate_manifest_sha256']


def test_gpu_cells_embed_adapter_for_old_locked_checkouts():
    root = Path(__file__).resolve().parents[1]
    book = nbformat.read(root / 'notebooks/04_colab_submit_available_data.ipynb', as_version=4)
    sources = [c.source for c in book.cells if c.cell_type == 'code']
    helper = (root / 'scripts/submission_gpu_progress.py').read_text(encoding='utf-8').strip()
    for source in sources[2:4]:
        ast.parse(source)
        assert helper in source
        assert 'with submission_gpu_progress(' in source
