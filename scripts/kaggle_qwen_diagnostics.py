"""Read small Qwen tuning sidecars from the exact pinned GPU result version."""
import json
import math
from pathlib import Path
import re

from vietmedbridge.artifacts import atomic_json, read_json, utc_now
from vietmedbridge.content_embeddings import checked, seal


FIELDS = ('policy', 'mode', 'active_mode', 'runtime_fallback', 'kernel', 'batch_cap',
          'max_padded_tokens', 'sample_texts', 'sample_tokens', 'sample_token_range',
          'baseline_seconds', 'selected_seconds', 'measured_sample_speedup',
          'baseline_texts_per_second', 'selected_texts_per_second',
          'oom_backoffs', 'trials', 'confirmation', 'scope')


def tuning_summary(value, expected_encoder):
    if (not isinstance(value, dict) or value.get('encoder') != expected_encoder
            or value.get('policy') not in ('qwen-token-batches-v1', 'qwen-token-batches-v2')
            or value.get('mode') not in ('native', 'token_batches')
            or value.get('kernel') not in ('native', 'fp16_math')
            or type(value.get('batch_cap')) is not int or value['batch_cap'] <= 0):
        raise ValueError('Tuning report differs from the expected Qwen encoder/execution policy.')
    for name in ('baseline_seconds', 'selected_seconds', 'measured_sample_speedup'):
        v = value.get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid measured Qwen tuning timings.')
    return {name: value[name] for name in FIELDS if name in value}


def read_qwen_tuning(transport, manifest, project, expected_encoder, *, report=print):
    """Diagnostics are independent of native vector validation and task coverage.

    A missing/invalid sidecar is recorded and does not discard valid GPU work.
    A completed summary survives Colab restarts and avoids another sidecar read.
    """
    manifest = checked(manifest)
    result = manifest['manifest_sha256']
    if not re.fullmatch('[0-9a-f]{64}', result):
        raise ValueError('Invalid completed result digest.')
    plan = manifest.get('plan', {})
    team = plan.get('team', {})
    account, gpu_count, accounts = manifest.get('account_id'), team.get('gpu_count'), team.get('accounts')
    if (plan.get('manifest_sha256') != manifest.get('execution_plan')
            or type(account) is not int or type(accounts) is not int
            or not 0 <= account < accounts or type(gpu_count) is not int or gpu_count != 2):
        raise ValueError('Tuning report source has no valid execution account assignment.')
    checked(plan)
    identity = {'result': result, 'pinned_handle': transport.pinned, 'job': manifest['job'],
                'execution_plan': manifest['execution_plan'], 'account_id': account,
                'encoder': expected_encoder}
    destination = Path(project)/'data_staging/kaggle_qwen_diagnostics'/(result+'.json')
    if destination.is_file():
        saved = checked(read_json(destination))
        if (all(saved.get(k) == v for k, v in identity.items())
                and saved.get('state') == 'QWEN_TUNING_REPORTS_SAVED'):
            report('QWEN_TUNING_REPORTS_REUSED: saved diagnostic JSON; no GPU or sidecar replay')
            for item in saved['workers']:
                report('QWEN_TUNING_REPORT: '+json.dumps(item, ensure_ascii=False, allow_nan=False))
            return saved
    workers = []
    for worker in range(account*gpu_count, (account+1)*gpu_count):
        relative = f'vmb_full_queue/result/qwen-tuning/worker-{worker}.json'
        try:
            path = Path(transport._download(path=relative))
            if not path.is_file() or path.stat().st_size > 262_144:
                raise ValueError('Qwen tuning sidecar is missing or unexpectedly large.')
            value = tuning_summary(read_json(path), expected_encoder)
            # Reject nonfinite nested trial values before writing the summary.
            json.dumps(value, allow_nan=False)
            item = {'worker_id': worker, 'state': 'READ', **value}
        except Exception as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            item = {'worker_id': worker, 'state': 'UNAVAILABLE',
                    'error_type': type(error).__name__, 'http_status': status}
            # Avoid exception text: HTTP client errors can contain signed URLs.
        workers.append(item)
        report('QWEN_TUNING_REPORT: '+json.dumps(item, ensure_ascii=False, allow_nan=False))
    saved = seal({'schema': 'qwen-tuning-diagnostics-v1', **identity, 'workers': workers,
                  'state': 'QWEN_TUNING_REPORTS_SAVED' if all(w['state'] == 'READ' for w in workers)
                          else 'QWEN_TUNING_REPORTS_INCOMPLETE',
                  'updated_at': utc_now(), 'tasks_committed': 0})
    atomic_json(destination, saved)
    report('QWEN_TUNING_REPORTS_ON_DRIVE: '+str(destination))
    return saved
