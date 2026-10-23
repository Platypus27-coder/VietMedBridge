"""A complete text library stays fixed while bounded GPU plans consume its queue."""
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq
import pytest

from test_new_data_kaggle import project, Encoder
from scripts.new_data_embeddings import FAMILIES
from scripts.new_data_kaggle import NewDataKaggle
from scripts.new_data_kaggle import download_outputs
from scripts.full_queue_embeddings import FullQueueKaggle, run_worker, finalize, validate_library
from vietmedbridge.artifacts import atomic_json, read_json
from vietmedbridge.content_embeddings import ContentBlocks, seal, key


def library(project, tmp_path, *, accounts=2):
    directory, identities, checkout = project
    flow = FullQueueKaggle(directory, tmp_path/'cpu', identities, accounts=accounts)
    target, job = flow.export_inputs(checkout, config={}, runtime={}, code_commit='a'*40)
    return flow, target, job


def run_account(directory, job, plan, output, work, identities, account_id, *, stop=False):
    calls = {family: [] for family in FAMILIES}
    factories = {family: lambda f=family: Encoder(identities[f], calls[f]) for family in FAMILIES}
    count = job['team']['gpu_count']
    for worker in range(account_id*count, (account_id+1)*count):
        run_worker(directory, plan, output/'result', work/f'worker-{worker}',
                   worker_id=worker, factories=factories,
                   deadline=time.monotonic()+(-1 if stop else 60))
    result = finalize(directory, plan, output, account_id)
    return calls, result


def resealed(value, **changes):
    return seal({**{k: v for k, v in value.items() if k != 'manifest_sha256'}, **changes})


def test_full_library_contains_every_pending_asset_despite_tiny_execution_budget(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    assert job['schema'] == 'full-data-text-library-v1'
    assert job['input_export_complete'] is True and job['unscheduled_tasks'] == 0
    selected = flow.select()
    expected = {task['selection_id'] for family in FAMILIES
                for entry in selected[family].values() for task in entry['tasks']}
    assert {task['id'] for task in job['tasks']} == expected
    assert len(job['assets']) < len(job['tasks'])  # source parts, not one file per GPU task
    for task in job['tasks']:
        assert (target/task['asset']).is_file()
        rows = pq.read_table(target/task['asset']).to_pylist()[task['offset']:task['offset']+task['rows']]
        assert len(rows) == task['rows']
        assert all(row['id'] == key(row['text']) and row['text'] != 'shared pre-existing' for row in rows)
    _, plan = flow.create_plan(job, output_bytes=2100)
    assert plan['schema'] == 'full-data-execution-plan-v1'
    assert 0 < len(plan['tasks']) < len(job['tasks'])
    assert all(value <= 2100 for value in plan['estimated_bytes_per_account'])


def test_independent_account_imports_and_repeated_plans_finish_one_immutable_library(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    before = (project[0]/'data/active_data_candidate.json').read_bytes()
    job_bytes = (target/'job.json').read_bytes()
    encoded = {family: [] for family in FAMILIES}
    plans, completed_ids = set(), set()
    for round_id in range(50):
        _, plan = flow.create_plan(job, output_bytes=2100)
        if plan.get('state') == 'NO_PENDING_KAGGLE_TASKS':
            break
        assert plan['manifest_sha256'] not in plans
        plans.add(plan['manifest_sha256'])
        assert not completed_ids & set(plan['tasks'])
        for account in range(2):
            output = tmp_path/f'round-{round_id}-account-{account}'
            calls, result = run_account(target, job, plan, output,
                                       tmp_path/f'gpu-{round_id}-{account}', project[1], account)
            for family in FAMILIES:
                assert not set(calls[family]) & set(encoded[family])
                encoded[family].extend(calls[family])
            report = flow.install(output/'result')
            completed_ids.update(task['task'] for task in result['tasks'])
    else:
        pytest.fail('The bounded plans did not drain a finite input library.')
    assert len(plans) > 1
    assert all(len(values) == 11 and len(set(values)) == 11 for values in encoded.values())
    assert all(value['completed_parts'] == value['total_parts'] == 10 for value in flow.coverage().values())
    assert (target/'job.json').read_bytes() == job_bytes
    assert (project[0]/'data/active_data_candidate.json').read_bytes() == before


def test_partial_deadline_and_account_import_keep_unfinished_tasks_pending(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, output_bytes=2100)
    output0 = tmp_path/'account-0'
    calls0, result0 = run_account(target, job, plan, output0, tmp_path/'gpu-0', project[1], 0, stop=True)
    assert not any(calls0.values())
    assert result0['tasks'] == [] and result0['state'] == 'CHECKPOINTED_PARTIAL'
    flow.install(output0/'result')
    _, waiting = flow.create_plan(job, output_bytes=2100)
    assert waiting['state'] == 'WAITING_FOR_KAGGLE_OUTPUTS'
    output1 = tmp_path/'account-1'
    _, result1 = run_account(target, job, plan, output1, tmp_path/'gpu-1', project[1], 1)
    flow.install(output1/'result')
    _, next_plan = flow.create_plan(job, output_bytes=2100)
    assert next_plan['manifest_sha256'] != plan['manifest_sha256']
    assert not {receipt['task'] for receipt in result1['tasks']} & set(next_plan['tasks'])
    assert any(task in next_plan['tasks'] for task in plan['tasks']
               if next(t for t in job['tasks'] if t['id'] == task)['worker'] < 2)


@pytest.mark.parametrize('change', ['foreign-library', 'duplicate-task', 'unknown-task', 'over-budget'])
def test_bad_execution_plan_rejected_before_model_construction(project, tmp_path, change):
    flow, target, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, output_bytes=2100)
    if change == 'foreign-library':
        bad = resealed(plan, library_job='f'*64)
    elif change == 'duplicate-task':
        bad = resealed(plan, tasks=plan['tasks']+[plan['tasks'][0]])
    elif change == 'unknown-task':
        bad = resealed(plan, tasks=plan['tasks']+['f'*64])
    else:
        bad = resealed(plan, output_bytes_per_account=1)
    def fail():
        pytest.fail('An invalid execution plan reached model construction.')
    with pytest.raises((ValueError, KeyError)):
        run_worker(target, bad, tmp_path/'result', tmp_path/'gpu', worker_id=0,
                   factories={family: fail for family in FAMILIES}, deadline=time.monotonic()+60)
    assert not list((tmp_path/'result').rglob('*.npy'))


def test_legacy_job_import_still_works_after_library_export(project, tmp_path):
    from scripts.new_data_kaggle import run_worker as legacy_worker, finalize as legacy_finalize
    directory, identities, checkout = project
    legacy = NewDataKaggle(directory, tmp_path/'legacy-cpu', identities, accounts=2)
    old_target, old_job = legacy.export(checkout, config={}, runtime={}, code_commit='a'*40, output_bytes=2100)
    atomic_json(legacy.root/'active_round.json', seal({'binding': legacy.binding,
                'job': old_job['manifest_sha256'], 'dataset_handle': 'owner/old-dataset'}))
    active = (legacy.root/'active_round.json').read_bytes()
    flow = FullQueueKaggle(directory, tmp_path/'full-cpu', identities, accounts=2)
    target, job = flow.export_inputs(checkout, config={}, runtime={}, code_commit='a'*40)
    assert (legacy.root/'active_round.json').read_bytes() == active
    assert checked_legacy_job(legacy, old_job) == old_job
    for account in range(2):
        output = tmp_path/f'legacy-account-{account}'
        factories = {family: lambda f=family: Encoder(identities[f], []) for family in FAMILIES}
        for worker in range(account*2, account*2+2):
            legacy_worker(old_target, output/'result', tmp_path/f'legacy-gpu-{worker}',
                          worker_id=worker, factories=factories, deadline=time.monotonic()+60)
        legacy_finalize(old_target, output, account)
        flow.install(output/'result')
    _, plan = flow.create_plan(job, output_bytes=2100)
    assert not {task['id'] for task in old_job['tasks']} & set(plan['tasks'])
    assert (legacy.root/'active_round.json').read_bytes() == active


def test_changed_task_receipts_and_vector_cannot_publish_full_queue_coverage(project, tmp_path):
    flow, target, job = library(project, tmp_path)
    _, plan = flow.create_plan(job)
    output = tmp_path/'account-0'
    _, result = run_account(target, job, plan, output, tmp_path/'gpu', project[1], 0)
    assert result['tasks'] and result['blocks']
    result_path = output/'result/new-data-result.json'
    forged = resealed(plan, cached_keys={family: ['f'*64] for family in FAMILIES})
    atomic_json(result_path, resealed(result, plan=forged))
    with pytest.raises(ValueError, match='saved full-library execution plan'):
        flow.install(output/'result')
    assert not flow.imported()
    atomic_json(result_path, resealed(result, tasks=result['tasks']+[result['tasks'][0]]))
    with pytest.raises(ValueError):
        flow.install(output/'result')
    assert not flow.imported()
    unassigned = next(task for task in job['tasks'] if task['worker'] >= 2)
    rows = pq.read_table(target/unassigned['asset']).to_pylist()
    keys = [row['id'] for row in rows[unassigned['offset']:unassigned['offset']+unassigned['rows']]]
    extra = {'task': unassigned['id'], 'keys': keys}
    atomic_json(result_path, resealed(result, tasks=result['tasks']+[extra]))
    with pytest.raises(ValueError):
        flow.install(output/'result')
    assert not flow.imported()
    atomic_json(result_path, result)
    (output/'result'/result['blocks'][0]['path']).write_bytes(b'corrupt vector')
    with pytest.raises(ValueError):
        flow.install(output/'result')
    assert not flow.imported()


def test_library_preflight_checks_all_shared_files_without_reading_texts_or_models(project, tmp_path, monkeypatch):
    _, target, job = library(project, tmp_path)
    original = Path.read_bytes
    def no_text_reads(path):
        if path.suffix == '.parquet':
            pytest.fail('Library startup inventory reopened a text payload.')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', no_text_reads)
    assert validate_library(target) == job
    missing = next(iter(job['assets']))
    (target/missing).unlink()
    with pytest.raises(FileNotFoundError, match='missing 1 shared text files'):
        validate_library(target)


def test_saved_library_reopens_no_frozen_inputs_or_text_assets(project, tmp_path, monkeypatch):
    flow, target, job = library(project, tmp_path)
    def fail(*a, **k):
        pytest.fail('Saved full-library export repeated text payload selection.')
    monkeypatch.setattr(pq, 'read_table', fail)
    again_target, again_job = flow.export_inputs(project[2], config={}, runtime={}, code_commit='a'*40,
                                               blocks_factory=fail)
    assert again_target == target and again_job == job


def test_cold_retry_keeps_the_exact_upload_pending_execution_plan(project, tmp_path, monkeypatch):
    flow, _, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, output_bytes=2100)
    fresh = FullQueueKaggle(project[0], tmp_path/'fresh-cpu', project[1], accounts=2)
    def fail(*a, **k):
        pytest.fail('Upload-pending retry repeated task-key selection or receipt inventory.')
    monkeypatch.setattr(pq, 'read_table', fail)
    target, resumed = fresh.create_plan(job, output_bytes=2100, blocks_factory=fail)
    assert resumed == plan and read_json(target/'job.json') == plan


def test_promoting_an_unstarted_legacy_round_requires_explicit_confirmation_and_preserves_it(project, tmp_path):
    directory, identities, checkout = project
    old = NewDataKaggle(directory, tmp_path/'old', identities, accounts=2)
    _, old_job = old.export(checkout, config={}, runtime={}, code_commit='a'*40, output_bytes=2100)
    active = seal({'binding': old.binding, 'job': old_job['manifest_sha256'], 'dataset_handle': 'owner/legacy'})
    atomic_json(old.root/'active_round.json', active)
    flow = FullQueueKaggle(directory, tmp_path/'full', identities, accounts=2)
    _, job = flow.export_inputs(checkout, config={}, runtime={}, code_commit='a'*40)
    target, blocked = flow.create_plan(job, output_bytes=2100)
    assert target is None and blocked['state'] == 'LEGACY_ROUND_REQUIRES_EXPLICIT_SUPERSEDE'
    target, plan = flow.create_plan(job, output_bytes=2100, supersede_unstarted_legacy=True)
    assert target.is_dir() and plan['schema'] == 'full-data-execution-plan-v1'
    assert read_json(old.root/'active_round.json') == active
    assert checked_legacy_job(old, old_job) == old_job
    history = read_json(old.root/'history'/('legacy-'+old_job['manifest_sha256']+'.json'))
    assert history['legacy'] == active and history['unstarted_confirmed'] is True


def test_import_rechecks_cache_skip_keys_in_the_exact_cpu_producer(project, tmp_path, monkeypatch):
    import scripts.new_data_kaggle as base
    flow, target, job = library(project, tmp_path)
    # A later trusted cache block becomes available after the immutable text
    # library was made; execution must omit it and import must recheck it.
    for family in FAMILIES:
        cache = ContentBlocks(project[0]/'data', Encoder(project[1][family], []), tmp_path/f'seed-{family}')
        cache.refs(['duplicate new text'], batch_size=2, encode_missing=True)
        cache.close()
    _, plan = flow.create_plan(job)
    assert any(plan['cached_keys'].values())
    output = tmp_path/'account-0'
    _, result = run_account(target, job, plan, output, tmp_path/'gpu', project[1], 0)
    assert result['tasks']
    prior_imports = flow.imported()
    checked_keys = []
    def missing_lookup(blocks, keys):
        checked_keys.extend(keys)
        return {}
    monkeypatch.setattr(base, 'lookup', missing_lookup)
    with pytest.raises(ValueError, match='cached vectors are not available'):
        flow.install(output/'result')
    assert checked_keys and flow.imported() == prior_imports


def test_saved_plan_filename_cannot_alias_another_sealed_plan(project, tmp_path):
    flow, _, job = library(project, tmp_path)
    _, plan = flow.create_plan(job, output_bytes=2100)
    changed = resealed(plan, created_at='changed-checkpoint')
    atomic_json(flow.root/'plans'/(plan['manifest_sha256']+'.json'), changed)
    with pytest.raises(ValueError, match='filename/hash binding'):
        flow.saved_plan(plan['manifest_sha256'])


def checked_legacy_job(flow, job):
    return read_json(flow.root/'exports'/('job-'+job['manifest_sha256']+'.json'))


def test_failed_kaggle_session_can_recover_saved_outputs_with_native_validation(project, tmp_path, monkeypatch):
    import vietmedbridge.kaggle_outputs as transport
    flow, target, job = library(project, tmp_path)
    _, plan = flow.create_plan(job)
    output = tmp_path/'saved-kaggle-version'
    _, result = run_account(target, job, plan, output, tmp_path/'gpu', project[1], 0)
    calls = []
    class API:
        def kernels_status(self, handle):
            return {'status': 'ERROR'}
    def saved_pages(api, handle, destination, file_pattern=None, **kwargs):
        calls.append(file_pattern)
        source = output/'result'
        paths = [source/'new-data-result.json'] if file_pattern else [p for p in source.rglob('*') if p.is_file()]
        copies = []
        for path in paths:
            copied = Path(destination)/'result'/path.relative_to(source)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copied)
            copies.append(copied)
        return copies
    monkeypatch.setattr(transport, '_download_pages', saved_pages)
    restored = download_outputs(API(), 'owner/saved-error', tmp_path/'restore', allow_failed=True)
    assert len(calls) == 2 and calls[0] and calls[1] is None
    assert read_json(restored/'new-data-result.json') == result
    flow.install(restored)
    assert {receipt['task'] for receipt in result['tasks']} <= flow.imported()


@pytest.mark.parametrize('status,allow_failed', [('RUNNING',True), ('QUEUED',True), ('ERROR',False)])
def test_running_or_unapproved_failed_session_is_rejected_before_output_download(tmp_path, monkeypatch, status, allow_failed):
    import vietmedbridge.kaggle_outputs as transport
    class API:
        def kernels_status(self, handle):
            return {'status': status}
    def unexpected_download(*args, **kwargs):
        pytest.fail('Non-terminal/unapproved session must not download or reuse stale outputs')
    monkeypatch.setattr(transport, '_download_pages', unexpected_download)
    with pytest.raises(RuntimeError, match='status='+status):
        download_outputs(API(), 'owner/live', tmp_path, allow_failed=allow_failed)
