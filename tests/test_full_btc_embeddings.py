"""Full membership gate, native input reuse, complete BGE inheritance and resume."""
from dataclasses import asdict
from pathlib import Path
import json

import pytest
import pyarrow.parquet as pq

from test_external_import import external, CharacterTokenizer, SPEC, CONFIG
from test_maximize_collection import collected
from scripts import full_btc_embeddings as full
from scripts.new_data_kaggle import NewDataKaggle
from vietmedbridge.artifacts import atomic_json, code_fingerprint, digest_json, read_json
from vietmedbridge.content_embeddings import checked, seal


def prepared(tmp_path, external):
    collection, flow, reader, links, body = collected(tmp_path, external)
    tokenizer = CharacterTokenizer()
    golden = {'passed': True, 'code_sha256': code_fingerprint(), 'tokenizer': SPEC, 'chunking': asdict(CONFIG)}
    flow.chunk_ready(tokenizer, phase='metadata-text-v1')
    baseline = flow.freeze_delta('metadata-text-v1', links, tokenizer, golden, audit_size=2)
    flow.recover_raw(reader)
    phase = flow.stage_ready_batch()
    flow.chunk_ready(tokenizer, phase=phase)
    recovered = flow.freeze_delta(phase, links, tokenizer, golden, audit_size=2)
    root = flow.root / 'processed/metadata-text-v1'
    candidate = read_json(root / baseline['candidate_name'])
    alias = candidate['health']['files']['canonical_aliases']
    label = (root / alias['path']).relative_to(collection.project).as_posix()
    collection.existing(root / alias['path'], alias['sha256'], label)
    # Model a real collection where existing IDs never enter the recovery queue.
    flow.con.execute('DELETE FROM aliases WHERE id IN (100,101)')
    flow.con.commit()
    flow.snapshot()
    one = full._source(collection.project, 'baseline', root.relative_to(collection.project).as_posix(),
                       baseline['candidate_name'], baseline['candidate_manifest_sha256'], mapping=label, alias_sha=alias['sha256'])
    raw_root = flow.root / 'processed' / phase
    two = full._source(collection.project, phase, raw_root.relative_to(collection.project).as_posix(),
                       recovered['candidate_name'], recovered['candidate_manifest_sha256'])
    contract = {'workflow': 'FULL_VALIDATED_BTC_EMBEDDINGS_V1', 'sources': [one, two],
                'collection_report': 'fixture', 'expected_reused': 2, 'official_total': 5,
                'collection_pointer': checked(read_json(collection.pointer)),
                'maximize_pointer': checked(read_json(flow.pointer)), 'scope': 'All prepared fixture sources'}
    return collection, flow, contract


def test_all_existing_and_recovered_ids_are_audited_and_cold_resume_skips_sql(tmp_path, external, monkeypatch):
    collection, flow, contract = prepared(tmp_path, external)
    monkeypatch.setattr(full, 'inventory', lambda *a: contract)
    root, report = full.prepare_full_scope(collection.project, tmp_path / 'checkout', tmp_path / 'audit', minimum_ids=3)
    assert report['official_ids'] == 3 and report['reused_ids'] == 2 and report['new_chunked_ids'] == 1
    assert report['source_builds'] == 2 and report['source_documents_sum'] == 3
    members = [row for part in report['membership']['parts'] for row in pq.read_table(root / part['path']).to_pylist()]
    assert {row['official_id'] for row in members} == {100, 101, 103}
    assert {row['source'] for row in members} == {'baseline', contract['sources'][1]['build_run']}
    # Failed/absent current BTC IDs cannot masquerade as prepared documents.
    assert not {102, 104} & {row['official_id'] for row in members}
    def fail(*a, **k): pytest.fail('Committed full scope replayed old SQL/corpus')
    monkeypatch.setattr(full, '_restore', fail)
    monkeypatch.setattr(full, 'inventory', fail)
    assert full.prepare_full_scope(collection.project, None, tmp_path / 'new-runtime', minimum_ids=3)[1] == report
    _, batch, sources = full.load_full_batch(collection.project, root.relative_to(collection.project).as_posix(), minimum_ids=3)
    assert batch['official_ids'] == 3 and len({s['build']['build_run'] for s in sources}) == 2
    flow.close(); collection.close()


def test_partial_scope_cannot_pass_requested_full_coverage(tmp_path, external, monkeypatch):
    collection, flow, contract = prepared(tmp_path, external)
    monkeypatch.setattr(full, 'inventory', lambda *a: contract)
    with pytest.raises(ValueError, match='required >= 4'):
        full.prepare_full_scope(collection.project, None, tmp_path / 'audit', minimum_ids=4)
    assert not list((collection.project / full.PARENT).glob('*/scope.json'))
    flow.close(); collection.close()


def test_membership_resumes_after_committed_part_without_rewriting_it(tmp_path, external, monkeypatch):
    collection, flow, contract = prepared(tmp_path, external)
    monkeypatch.setattr(full, 'inventory', lambda *a: contract)
    original = full.atomic_json
    def interrupt(path, value):
        original(path, value)
        if Path(path).name == 'membership_progress.json':
            raise KeyboardInterrupt('Stopped after durable membership part')
    with monkeypatch.context() as patch:
        patch.setattr(full, 'atomic_json', interrupt)
        with pytest.raises(KeyboardInterrupt):
            full.prepare_full_scope(collection.project, None, tmp_path / 'audit', minimum_ids=3)
    def fail(*a, **k): pytest.fail('Committed membership payload was rewritten')
    monkeypatch.setattr(full, 'publish_file', fail)
    root, report = full.prepare_full_scope(collection.project, None, tmp_path / 'audit', minimum_ids=3)
    assert report['official_ids'] == 3 and len(report['membership']['parts']) == 1
    flow.close(); collection.close()


def test_alias_provenance_cannot_point_at_an_unregistered_source(tmp_path, external, monkeypatch):
    collection, flow, contract = prepared(tmp_path, external)
    contract['sources'][0]['mapping_sha256'] = '0' * 64
    monkeypatch.setattr(full, 'inventory', lambda *a: contract)
    with pytest.raises(ValueError, match='alias provenance'):
        full.prepare_full_scope(collection.project, None, tmp_path / 'audit', minimum_ids=3)
    flow.close(); collection.close()


def test_inherited_complete_bge_skips_input_and_never_rewrites_native_view(tmp_path, external, monkeypatch):
    collection, flow, contract = prepared(tmp_path, external)
    identities = {family: {'encoder': {'model_id': family, 'revision': 'test'}, 'dimension': 3} for family in ('bge', 'qwen')}
    record = contract['sources'][0]
    inputs = checked(read_json(collection.project / record['build_dir'] / 'index_inputs/units.json'))
    native = collection.project / 'data/native-complete-bge'
    identity = {'input_manifest_sha256': inputs['manifest_sha256'], 'input_count': inputs['input_count'],
                **identities['bge'], 'format': 'content-reference-v1'}
    parts = [seal({'signature': digest_json(identity), 'start': p['start'], 'rows': p['rows'], 'input_sha256': p['sha256'],
                   'path': f"map-{p['start']:012d}.parquet", 'sha256': 'a'*64}) for p in inputs['parts']]
    complete = seal({**identity, 'signature': digest_json(identity), 'state': 'COMPLETE', 'parts': parts})
    atomic_json(native / 'embeddings.json', complete)
    record['completed_views']['bge'] = {'path': native.relative_to(collection.project).as_posix(),
                                       'manifest_sha256': complete['manifest_sha256']}
    monkeypatch.setattr(full, 'inventory', lambda *a: contract)
    root, report = full.prepare_full_scope(collection.project, None, tmp_path / 'audit', minimum_ids=3)
    original = (native / 'embeddings.json').read_bytes()
    runner = NewDataKaggle(collection.project, tmp_path / 'cpu', identities, accounts=2,
        batch_loader=lambda p: full.load_full_batch(p, root.relative_to(p).as_posix(), minimum_ids=3))
    baseline_sources = [s for s in runner.view.sources if s['build']['build_run'] == 'baseline']
    assert len(runner.view._journals('bge')) == len(baseline_sources)
    # No vector/map files exist in the native view: inheritance must be metadata-only.
    assert list(native.iterdir()) == [native / 'embeddings.json']
    import scripts.new_data_kaggle as pipeline
    real_publish = pipeline.publish_file
    staged = []
    def record_input(source, destination):
        if str(destination).endswith('current-input.parquet'): staged.append(Path(source))
        return real_publish(source, destination)
    monkeypatch.setattr(pipeline, 'publish_file', record_input)
    runner.select()
    baseline_paths = [collection.project / s['build']['build_dir'] / 'index_inputs' / s['part']['path'] for s in baseline_sources]
    assert all(staged.count(path) == 1 for path in baseline_paths)  # Qwen only; BGE paid work skipped.
    assert (native / 'embeddings.json').read_bytes() == original
    checkout = tmp_path / 'checkout'
    for name in ('retrieval_models.py', 'qwen_models.py'):
        path = checkout / 'src/vietmedbridge' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    target, job = runner.export(checkout, config={}, runtime={}, code_commit='a'*40)
    assert job['coverage_scope']['official_ids'] == 3
    assert job['coverage_scope']['source_builds'] == 2
    assert len(job['frozen_sources']) == 2
    assert job['frozen_sources'][0]['inherited_complete_models'] == ['bge']
    assert job['all_remaining_tasks_scheduled'] and job['unscheduled_tasks'] == 0
    assert any(t['family'] == 'qwen' for t in job['tasks'])
    flow.close(); collection.close()


def test_full_notebook_is_gated_and_embeds_the_full_scope_loader():
    import ast
    from scripts.write_new_data_kaggle_notebooks import full_transfer
    book = full_transfer()
    for cell in book.cells:
        if cell.cell_type == 'code': compile(cell.source, '<full-export>', 'exec')
    assert 'minimum_ids=802028' in book.cells[2].source
    assert 'prepare_full_scope(' in book.cells[2].source
    assert 'load_full_batch(' in book.cells[2].source
    assert 'full_flow(' in book.cells[3].source and 'recovery_flow(' not in book.cells[3].source
    # Old input text/parser/model code is not modified to create a new cache namespace.
    assert "code_commit=lock['commit']" in book.cells[3].source
