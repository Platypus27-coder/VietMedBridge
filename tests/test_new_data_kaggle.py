"""Exercise sparse export -> two GPU assignments -> import -> cold continuation."""
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.new_data_embeddings import PARENT, FAMILIES, load_new_batch
from scripts.new_data_kaggle import NewDataKaggle, run_worker, finalize, validate_job
from scripts.new_data_cache_lookup import ResumableContentBlocks
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.content_embeddings import ContentBlocks, key, seal, checked
import scripts.new_data_kaggle as pipeline


class Encoder:
    def __init__(self, expected, calls):
        self.identity, self.dimension, self.calls = expected['encoder'], expected['dimension'], calls

    def encode(self, texts, *, batch_size):
        self.calls.extend(texts)
        values = np.array([[1+int(key(t)[i:i+2], 16) for i in range(0, self.dimension*2, 2)] for t in texts], dtype=np.float32)
        return values/np.linalg.norm(values, axis=1, keepdims=True)

    def close(self):
        pass


@pytest.fixture
def project(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('p')
    project = tmp_path/'project'
    root = project/'data_staging'/PARENT/'freeze_batch_v1'
    builds, sources = [], []
    for i, name in enumerate([f'new-{i:05d}' for i in range(9)]+['retained-71be0e12']):
        build = root/'builds'/name
        texts = ['shared pre-existing', f'new text {i}', 'duplicate new text']
        path = build/'index_inputs/part-0.parquet'
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{'id': key(t), 'text': t} for t in texts]), path)
        part = {'start': 0, 'rows': len(texts), 'signature': digest_json(name), 'path': path.name, 'sha256': sha256_file(path)}
        inputs = seal({'state': 'COMPLETE', 'candidate_manifest_sha256': digest_json(['candidate', name]),
                       'input_count': len(texts), 'signature': digest_json(name), 'parts': [part]})
        atomic_json(build/'index_inputs/units.json', inputs)
        source = {'build_run': name, 'build_dir': build.relative_to(project).as_posix(),
                  'snapshot_sha256': digest_json(['snapshot', name]), 'counts': {'documents': 1}}
        sources.append(source)
        builds.append({**source, 'state': 'COMPLETE', 'index_inputs_manifest_sha256': inputs['manifest_sha256'],
            'candidate_manifest_sha256': inputs['candidate_manifest_sha256'], 'unique_dense_inputs': 3, 'input_parts': 1})
    contract = seal({'sources': sources})
    atomic_json(root/'contract.json', contract)
    builds = [seal({**b, 'contract_sha256': contract['manifest_sha256']}) for b in builds]
    atomic_json(root/'freeze_batch.json', seal({'state': 'NEW_DATA_INPUTS_READY', 'completed_builds': 10, 'total_builds': 10,
        'contract_sha256': contract['manifest_sha256'], 'builds': builds, 'input_parts': 10}))
    identities = {f: {'encoder': {'model_id': f, 'revision': 'test'}, 'dimension': 3} for f in FAMILIES}
    atomic_json(project/'data/active_data_candidate.json', {'untouched': True})
    checkout = tmp_path/'checkout'
    for name in ('retrieval_models.py', 'qwen_models.py'):
        p = checkout/'src/vietmedbridge'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('fixture')
    for family in FAMILIES:
        enc = Encoder(identities[family], [])
        cache = ContentBlocks(project/'data', enc, tmp_path/'seed')
        cache.refs(['shared pre-existing'], batch_size=2, encode_missing=True)
        cache.close()
    return project, identities, checkout


def exported(project, tmp_path, *, accounts=1, output_bytes=16_000_000_000):
    directory, identities, checkout = project
    flow = NewDataKaggle(directory, tmp_path/'cpu', identities, accounts=accounts)
    target, job = flow.export(checkout, config={}, runtime={}, code_commit='a'*40, output_bytes=output_bytes)
    return flow, target, job


def execute(target, job, output, tmp_path, identities, *, stop=False):
    calls = {f: [] for f in FAMILIES}
    factories = {f: lambda f=f: Encoder(identities[f], calls[f]) for f in FAMILIES}
    for worker in range(job['team']['accounts']*2):
        run_worker(target, output/'result', tmp_path/f'gpu-{worker}', worker_id=worker,
                   factories=factories, deadline=time.monotonic()+(-1 if stop else 60))
    return calls


def test_real_sparse_two_gpu_round_reuses_old_and_duplicate_texts_and_finishes_maps(project, tmp_path, monkeypatch):
    flow, target, job = exported(project, tmp_path)
    pointer = (project[0]/'data/active_data_candidate.json').read_bytes()
    assert len(job['tasks']) > 0
    assert all('shared pre-existing' not in pq.read_table(target/t['asset']).column('text').to_pylist() for t in job['tasks'])
    output = tmp_path/'kaggle-output'
    calls = execute(target, job, output, tmp_path, project[1])
    # Identical text belongs to one GPU and is encoded once across all sources.
    assert all(len(calls[f]) == 11 and len(set(calls[f])) == 11 for f in FAMILIES)
    result = finalize(target, output, 0)
    assert result['state'] == 'ROUND_COMPLETE'
    assert len(list(output.rglob('*.npy'))) == len(result['blocks'])  # no second copy of outputs
    report = flow.install(output/'result')
    assert all(v['completed_parts'] == 10 for v in report['coverage'].values())
    assert (project[0]/'data/active_data_candidate.json').read_bytes() == pointer
    # A new CPU runtime only reads sealed journals, not source parts/maps/vectors.
    fresh = NewDataKaggle(project[0], tmp_path/'fresh-cpu', project[1], accounts=1)
    def fail(*args, **kwargs):
        pytest.fail('A completed CPU selection reopened payloads or inventoried receipts.')
    monkeypatch.setattr(pq, 'read_table', fail)
    target, report = fresh.export(project[2], config={}, runtime={}, code_commit='a'*40, blocks_factory=fail)
    assert target is None and report['state'] == 'NO_PENDING_KAGGLE_TASKS'


def test_selection_cold_resume_no_source_reads_and_no_gpu_weights(project, tmp_path, monkeypatch):
    flow, target, job = exported(project, tmp_path)
    def fail(*args, **kwargs):
        pytest.fail('Completed selection reopened frozen corpus inputs.')
    fresh = NewDataKaggle(project[0], tmp_path/'fresh', project[1], accounts=1)
    monkeypatch.setattr(pq, 'read_table', fail)
    selected = fresh.select(blocks_factory=fail)
    assert all(len(selected[f]) == 10 for f in FAMILIES)


def test_partial_capacity_import_then_next_round_skips_completed_and_cached_duplicates(project, tmp_path):
    flow, target, job = exported(project, tmp_path, output_bytes=2100)
    output = tmp_path/'round1'
    execute(target, job, output, tmp_path, project[1])
    finalize(target, output, 0)
    flow.install(output/'result')
    done = {t['selection_id'] for t in job['tasks']}
    target2, job2 = flow.export(project[2], config={}, runtime={}, code_commit='a'*40, output_bytes=2100)
    assert target2 is not None
    assert not done & {t['selection_id'] for t in job2['tasks']}
    old_keys = {k for t in read_json(output/'result/new-data-result.json')['tasks'] for k in t['keys']}
    for task in job2['tasks']:
        assert not set(pq.read_table(target2/task['asset']).column('id').to_pylist()) & old_keys


def test_deadline_saves_empty_partial_without_loading_models(project, tmp_path):
    flow, target, job = exported(project, tmp_path)
    output = tmp_path/'timeout'
    calls = execute(target, job, output, tmp_path, project[1], stop=True)
    result = finalize(target, output, 0)
    assert result['state'] == 'CHECKPOINTED_PARTIAL' and result['tasks'] == [] and result['blocks'] == []
    assert not any(calls.values())


def test_tampered_task_and_vector_cannot_publish_coverage(project, tmp_path):
    flow, target, job = exported(project, tmp_path)
    output = tmp_path/'bad'
    execute(target, job, output, tmp_path, project[1])
    result = finalize(target, output, 0)
    path = output/'result'/result['blocks'][0]['path']
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Missing or changed'):
        flow.install(output/'result')
    assert not flow.imported()


def test_two_accounts_assign_hash_ownership_consistently(project, tmp_path):
    flow, target, job = exported(project, tmp_path, accounts=2)
    owners = {}
    for t in job['tasks']:
        for text_key in pq.read_table(target/t['asset']).column('id').to_pylist():
            owner = owners.setdefault((t['family'], text_key), t['worker'])
            assert owner == t['worker'] == int(text_key[:8], 16) % 4
    assert {t['worker'] for t in job['tasks']} == {0, 1, 2, 3}


def test_two_accounts_import_independently_then_merge_complete(project, tmp_path, monkeypatch):
    flow, target, job = exported(project, tmp_path, accounts=2)
    for account in range(2):
        output = tmp_path/f'account-{account}'
        calls = {f: [] for f in FAMILIES}
        factories = {f: lambda f=f: Encoder(project[1][f], calls[f]) for f in FAMILIES}
        for worker in range(account*2, (account+1)*2):
            run_worker(target, output/'result', tmp_path/f'gpu-{worker}', worker_id=worker,
                       factories=factories, deadline=time.monotonic()+60)
        finalize(target, output, account)
        report = flow.install(output/'result')
    assert all(v['completed_parts'] == 10 for v in report['coverage'].values())
    # Idempotent re-import trusts its committed receipt and does not rehash vectors.
    def fail(*args, **kwargs):
        pytest.fail('A committed import re-opened vector payloads.')
    monkeypatch.setattr(pipeline, 'verify_file', fail)
    report = flow.install(output/'result')
    assert report['payloads_reopened'] is False


def test_wrong_encoder_identity_fails_before_new_payloads(project, tmp_path):
    flow, target, job = exported(project, tmp_path)
    bad = {f: lambda f=f: Encoder({'encoder': {'wrong': True}, 'dimension': 3}, []) for f in FAMILIES}
    worker = job['tasks'][0]['worker']
    with pytest.raises(ValueError, match='GPU model identity'):
        run_worker(target, tmp_path/'bad-model', tmp_path/'bad-work', worker_id=worker,
                   factories=bad, deadline=time.monotonic()+60)
    assert not list((tmp_path/'bad-model').rglob('*.npy'))


def test_generated_notebooks_compile_and_keep_cpu_gpu_entrypoints_separate():
    from scripts.write_new_data_kaggle_notebooks import transfer, gpu, NAMES
    cpu = transfer()
    for book in (cpu, gpu(0), gpu(1)):
        for c in book.cells:
            if c.cell_type == 'code':
                compile(c.source, '<notebook>', 'exec')
    source = ast.parse(cpu.cells[2].source)
    payload = next(n.value for n in source.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets))
    embedded = json.loads(ast.literal_eval(payload.args[0]))
    scripts = Path(pipeline.__file__).parent
    assert set(embedded) == set(NAMES)
    assert all(embedded[n] == (scripts/n).read_text(encoding='utf-8') for n in NAMES)
    assert all('drive.mount(' not in c.source for c in gpu().cells if c.cell_type == 'code')
    assert "KAGGLE_ACCOUNT_ID = 1" in gpu(1).cells[1].source
    assert "KAGGLE_ACCOUNTS = 2" in cpu.cells[1].source


def test_uploaded_round_waits_for_other_account_without_opening_payloads(project, tmp_path, monkeypatch):
    flow, target, job = exported(project, tmp_path, accounts=2)
    atomic_json(flow.root/'active_round.json', seal({'binding': flow.binding,
        'job': job['manifest_sha256'], 'dataset_handle': 'owner/private-new-data'}))
    def fail(*args, **kwargs):
        pytest.fail('An unsettled uploaded round restarted CPU selection.')
    monkeypatch.setattr(flow, 'select', fail)
    target, status = flow.export(project[2], config={}, runtime={}, code_commit='a'*40)
    assert target is None and status['state'] == 'WAITING_FOR_KAGGLE_OUTPUTS'
    assert status['imported_accounts'] == [] and status['required_accounts'] == [0, 1]


def test_worker_never_overwrites_outputs_from_a_previous_job(project, tmp_path):
    flow, target, job = exported(project, tmp_path)
    output = tmp_path/'existing-output'
    atomic_json(output/'job.json', {'old_paid_job': True})
    with pytest.raises(ValueError, match='another job'):
        run_worker(target, output, tmp_path/'gpu', worker_id=0, factories={}, deadline=time.monotonic()+60)
    assert read_json(output/'job.json') == {'old_paid_job': True}
