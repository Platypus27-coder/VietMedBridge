"""Inspect an uploaded round using local Kaggle metadata before loading weights.

This proves task-file presence and job accounting, not semantic text quality.
Native workers still check each payload SHA and text key when consuming it.
The source is embedded into notebooks so existing uploaded jobs stay immutable.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def inspect_job_inputs(directory, job, *, expected_scope=None, report=print):
    directory = Path(directory)
    digest = hashlib.sha256(json.dumps(
        {k: v for k, v in job.items() if k != 'manifest_sha256'},
        sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if job.get('manifest_sha256') != digest:
        raise ValueError('Job header changed; do not load model weights.')
    team = job.get('team', {})
    accounts, gpus = team.get('accounts'), team.get('gpu_count')
    if type(accounts) is not int or accounts < 1 or type(gpus) is not int or gpus < 1:
        raise ValueError('Invalid account/GPU assignment.')
    if expected_scope is not None:
        scope = job.get('coverage_scope', {})
        sources = job.get('frozen_sources', [])
        if (scope.get('scope_manifest_sha256') != expected_scope
                or type(scope.get('official_ids')) is not int or scope['official_ids'] < 802028
                or scope.get('source_builds') != 13 or len(sources) != 13
                or len({s.get('build_run') for s in sources}) != 13):
            raise ValueError('This is not the verified 802,028-ID / 13-source full-data scope.')
    tasks = job.get('tasks', [])
    if not tasks or job.get('scheduled_tasks') != len(tasks):
        raise ValueError('Scheduled task count differs from the exported round.')
    remaining = job.get('remaining_tasks_before_round', {})
    queued = job.get('unscheduled_tasks')
    if (type(queued) is not int or queued < 0
            or any(type(v) is not int or v < 0 for v in remaining.values())
            or sum(remaining.values()) != len(tasks) + queued
            or job.get('all_remaining_tasks_scheduled') is not (queued == 0)):
        raise ValueError('Pending task accounting changed; no GPU work started.')
    files = {p.name for p in directory.iterdir() if p.is_file()}
    seen, missing = set(), []
    by_model, by_account = {}, {}
    actual_tasks = Counter()
    for task in tasks:
        task_id, family, worker, rows = (task.get(k) for k in ('id', 'family', 'worker', 'rows'))
        if (not isinstance(task_id, str) or not re.fullmatch('[0-9a-f]{64}', task_id)
                or task_id in seen or task.get('selection_id') != task_id
                or family not in job.get('models', {})
                or type(worker) is not int or not 0 <= worker < accounts*gpus
                or type(rows) is not int or not 1 <= rows <= 512
                or task.get('asset') != 'task-'+task_id+'.parquet'
                or not re.fullmatch('[0-9a-f]{64}', task.get('sha256', ''))):
            raise ValueError('Invalid or duplicate task in uploaded round.')
        seen.add(task_id)
        if task['asset'] not in files:
            missing.append(task['asset'])
        actual_tasks[family] += 1
        for groups, key in ((by_model, family), (by_account, worker//gpus)):
            counts = groups.setdefault(key, {'tasks': 0, 'text_rows': 0})
            counts['tasks'] += 1
            counts['text_rows'] += rows
    if any(actual_tasks[k] > remaining.get(k, -1) for k in actual_tasks):
        raise ValueError('Model task count exceeds the retained queue.')
    if missing:
        raise FileNotFoundError(f'Uploaded dataset is missing {len(missing)} task files; '
                                f'no model weights loaded. Examples: {missing[:3]}')
    for name, checksum in job.get('worker_sources', {}).items():
        if name not in files or Path(name).name != name:
            raise FileNotFoundError('Uploaded worker source is missing: '+name)
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Uploaded worker source changed: '+name)
    summary = {'state': 'INPUT_INVENTORY_COMPLETE', 'job': job['manifest_sha256'],
               'scope': job.get('coverage_scope', {}), 'team': team, 'tasks': len(tasks),
               'queued_tasks_on_drive': queued, 'by_model': by_model, 'by_account': by_account,
               'payload_verification': 'SHA256_AND_TEXT_KEYS_ON_USE',
               'semantic_quality': 'NOT_CERTIFIED_BY_FILE_INVENTORY'}
    report('INPUT_INVENTORY_COMPLETE: all '+str(len(tasks))+' task files present; no model weights loaded.')
    report(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def inspect_full_queue_inputs(directory, library, plan, *, expected_scope, report=print):
    """Inventory the complete text library separately from the bounded GPU plan."""
    def checked(value):
        checksum = hashlib.sha256(json.dumps(
            {k: v for k, v in value.items() if k != 'manifest_sha256'},
            sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if value.get('manifest_sha256') != checksum:
            raise ValueError('Full-input library or execution plan header changed.')
        return value
    checked(library); checked(plan)
    if library.get('schema') != 'full-data-text-library-v1' or plan.get('schema') != 'full-data-execution-plan-v1':
        raise ValueError('Add one all-text library and one matching execution-plan dataset.')
    scope = library.get('coverage_scope', {})
    sources = library.get('frozen_sources', [])
    if (scope.get('scope_manifest_sha256') != expected_scope or scope.get('official_ids', 0) < 802028
            or scope.get('source_builds') != 13 or len(sources) != 13
            or len({s.get('build_run') for s in sources}) != 13
            or library.get('input_export_complete') is not True):
        raise ValueError('The full-input library does not cover the verified 13-source scope.')
    if plan.get('library_job') != library['manifest_sha256'] or plan.get('team') != library.get('team'):
        raise ValueError('Execution plan belongs to another text library or GPU team.')
    directory = Path(directory)
    files = {p.name for p in directory.iterdir() if p.is_file()}
    by_model, tasks, missing = {}, {}, set()
    for task in library.get('tasks', []):
        task_id = task.get('id')
        asset = task.get('asset', '')
        if (not isinstance(task_id, str) or not re.fullmatch('[0-9a-f]{64}', task_id)
                or task_id in tasks or task.get('selection_id') != task_id
                or not asset or Path(asset).name != asset or '\\' in asset
                or task.get('family') not in library.get('models', {})
                or type(task.get('rows')) is not int or not 1 <= task['rows'] <= 512
                or type(task.get('offset')) is not int or task['offset'] < 0):
            raise ValueError('Invalid task in all-input library.')
        tasks[task_id] = task
        if asset not in files:
            missing.add(asset)
        counts = by_model.setdefault(task['family'], {'tasks': 0, 'text_rows': 0})
        counts['tasks'] += 1; counts['text_rows'] += task['rows']
    if not tasks or library.get('unscheduled_tasks') != 0 or library.get('scheduled_tasks') != len(tasks):
        raise ValueError('All-input library task accounting is incomplete.')
    if missing:
        raise FileNotFoundError(f'Full-input dataset is missing {len(missing)} shared text files: {sorted(missing)[:3]}')
    planned = plan.get('tasks', [])
    if len(set(planned)) != len(planned) or not set(planned) <= set(tasks):
        raise ValueError('Execution plan contains duplicate or foreign tasks.')
    for name, checksum in library.get('worker_sources', {}).items():
        if name not in files or Path(name).name != name:
            raise FileNotFoundError('Uploaded worker source is missing: '+name)
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Uploaded worker source changed: '+name)
    result = {'state': 'ALL_TEXT_INPUTS_PRESENT', 'library_job': library['manifest_sha256'],
              'scope': scope, 'all_exported_tasks': len(tasks), 'shared_text_files': len({t['asset'] for t in tasks.values()}),
              'by_model': by_model, 'this_gpu_plan_tasks': len(planned),
              'input_export_complete': True, 'payload_verification': 'SHA256_AND_TEXT_KEYS_ON_USE',
              'semantic_quality': 'NOT_CERTIFIED_BY_FILE_INVENTORY'}
    report('ALL_TEXT_INPUTS_PRESENT: entire library present; GPU execution is planned separately.')
    report(json.dumps(result, ensure_ascii=False, indent=2))
    return result
