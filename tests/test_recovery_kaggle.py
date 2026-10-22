"""Frozen recovery manifests use the real sparse export/worker/import path."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_new_data_kaggle import project
from test_external_import import external
from scripts.new_data_kaggle import NewDataKaggle, finalize
from scripts.recovery_kaggle_batch import CAMPAIGN, load_recovery_batch
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.content_embeddings import key, seal


def recovery_project(project):
    directory, identities, checkout = project
    root = directory / CAMPAIGN
    builds = []
    for i, name in enumerate(['metadata-text-v1', 'raw-ready-'+'a'*16]):
        build = root / 'processed' / name
        text = ['shared pre-existing', f'recovery unique {i}', 'shared recovery text']
        part = build / 'index_inputs/part-0.parquet'
        part.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{'id': key(t), 'text': t} for t in text]), part)
        candidate = {'state': 'FROZEN_CANDIDATE', 'snapshot_sha256': digest_json(['snapshot', name]),
                     'counts': {'documents': 1}, 'golden': {'passed': True},
                     'health': {'integrity': {'passed': True, 'official_membership': 'VERIFIED'}}}
        candidate['candidate_manifest_sha256'] = digest_json(candidate)
        candidate_name = 'candidate-'+candidate['candidate_manifest_sha256'][:16]+'.json'
        atomic_json(build / candidate_name, candidate)
        inputs = seal({'state': 'COMPLETE', 'candidate_manifest_sha256': candidate['candidate_manifest_sha256'],
                       'input_count': 3, 'signature': digest_json(name), 'parts': [
                           {'start': 0, 'rows': 3, 'signature': digest_json(name), 'path': part.name,
                            'sha256': sha256_file(part)}]})
        atomic_json(build / 'index_inputs/units.json', inputs)
        builds.append({'state': 'DELTA_READY_FOR_EMBEDDING', 'data_root': CAMPAIGN, 'build_run': name,
                       'candidate_name': candidate_name, 'candidate_manifest_sha256': candidate['candidate_manifest_sha256'],
                       'snapshot_sha256': candidate['snapshot_sha256'], 'index_inputs_manifest_sha256': inputs['manifest_sha256'],
                       'unique_dense_inputs': 3, 'documents': 1, 'submission_candidate_changed': False})
    atomic_json(root / 'delta_candidates.json', seal({'builds': builds, 'submission_candidate_changed': False}))
    return directory, identities, checkout


def test_two_recovery_batches_export_encode_import_and_resume_without_payload_rescan(project, tmp_path, monkeypatch):
    directory, identities, checkout = recovery_project(project)
    before = (directory / 'data/active_data_candidate.json').read_bytes()
    flow = NewDataKaggle(directory, tmp_path / 'cpu', identities, accounts=2, batch_loader=load_recovery_batch)
    assert flow.view.batch['documents'] == 2 and len(flow.sources) == 2
    target, job = flow.export(checkout, config={}, runtime={}, code_commit='a'*40)
    assert job['source_batch_dir'] == flow.view.root.relative_to(directory).as_posix()
    for task in job['tasks']:
        assert 'shared pre-existing' not in pq.read_table(target / task['asset']).column('text').to_pylist()
    for account in range(2):
        from scripts.new_data_kaggle import run_worker
        from test_new_data_kaggle import Encoder
        import time
        output = tmp_path / ('account-'+str(account))
        calls = {family: [] for family in identities}
        factories = {family: lambda f=family: Encoder(identities[f], calls[f]) for family in identities}
        for worker in range(account*2, account*2+2):
            run_worker(target, output / 'result', tmp_path / ('gpu-'+str(worker)), worker_id=worker,
                       factories=factories, deadline=time.monotonic()+60)
        assert all(len(set(values)) == len(values) for values in calls.values())
        finalize(target, output, account)
        report = flow.install(output / 'result')
    assert all(value['completed_parts'] == 2 for value in report['coverage'].values())
    assert (directory / 'data/active_data_candidate.json').read_bytes() == before
    # An already exported batch remains bound even if the live campaign manifest changes.
    atomic_json(directory / CAMPAIGN / 'delta_candidates.json', seal({'builds': [], 'submission_candidate_changed': False}))
    fresh = NewDataKaggle(directory, tmp_path / 'fresh', identities, accounts=2,
        batch_loader=lambda p: load_recovery_batch(p, job['source_batch_dir']))
    def fail(*a, **k):
        pytest.fail('Completed selection reread model inputs or receipt inventory')
    monkeypatch.setattr(pq, 'read_table', fail)
    assert all(len(v) == 2 for v in fresh.select(blocks_factory=fail).values())


def test_changed_candidate_cannot_be_exported(project):
    directory, _, _ = recovery_project(project)
    candidate = next((directory / CAMPAIGN / 'processed/metadata-text-v1').glob('candidate-*.json'))
    value = read_json(candidate)
    value['counts']['documents'] = 999
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match='validated frozen snapshot'):
        load_recovery_batch(directory)


def test_actual_maximize_freeze_and_native_inputs_are_accepted(tmp_path, external):
    from dataclasses import asdict
    from test_maximize_collection import collected
    from test_external_import import CharacterTokenizer, SPEC, CONFIG
    from vietmedbridge.artifacts import code_fingerprint
    collection, flow, reader, links, body = collected(tmp_path, external)
    tokenizer = CharacterTokenizer()
    golden = {'passed': True, 'code_sha256': code_fingerprint(), 'tokenizer': SPEC, 'chunking': asdict(CONFIG)}
    flow.chunk_ready(tokenizer, phase='metadata-text-v1')
    flow.freeze_delta('metadata-text-v1', links, tokenizer, golden, audit_size=2)
    flow.recover_raw(reader)
    phase = flow.stage_ready_batch()
    flow.chunk_ready(tokenizer, phase=phase)
    flow.freeze_delta(phase, links, tokenizer, golden, audit_size=2)
    root, batch, sources = load_recovery_batch(collection.project)
    assert batch['documents'] == 3 and len(batch['builds']) == 2
    assert batch['input_rows_before_cross_batch_dedup'] == sum(b['unique_dense_inputs'] for b in batch['builds'])
    for source in sources:
        path = collection.project / source['build']['build_dir'] / 'index_inputs' / source['part']['path']
        rows = pq.read_table(path).to_pylist()
        assert all(key(row['text']) == row['id'] for row in rows)
    flow.close()
    collection.close()


def test_recovery_notebook_has_locked_sources_and_no_raw_recovery_execution():
    import ast
    from scripts.write_new_data_kaggle_notebooks import recovery_transfer, NAMES
    book = recovery_transfer()
    for cell in book.cells:
        if cell.cell_type == 'code': compile(cell.source, '<recovery-export>', 'exec')
    assert 'intake_code_lock.json' not in book.cells[1].source
    assert 'delta_candidates.json' in book.cells[1].source
    assert 'bbf09ec01d542a5b06a4c2beec245571abeb3515' in book.cells[1].source
    assert 'recovery_flow(' in book.cells[-1].source
    tree = ast.parse(book.cells[2].source)
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)}
    sources = json.loads(ast.literal_eval(assignments['SOURCES'].args[0]))
    assert set(sources) == set(NAMES)
    assert ast.literal_eval(assignments['RECOVERY_SOURCE']) == Path('scripts/recovery_kaggle_batch.py').read_text(encoding='utf-8')
