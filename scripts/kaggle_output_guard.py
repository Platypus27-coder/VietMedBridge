"""Keep verbose locked worker output off Jupyter's IOPub channel.

The original dataset worker, plan and vector files stay unchanged. Its complete
stdout/stderr is saved in Kaggle Outputs; the notebook receives bounded progress
messages and a small result summary instead of millions of text-hash lines.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time


_PROGRESS_PREFIXES = (
    'GPU worker ', 'KAGGLE_CHECKPOINT ', 'Qwen corpus execution:',
    'BAAI/bge-m3 | saved mini-block ',
    'Qwen/Qwen3-Embedding-8B | saved mini-block ',
)


def _read_progress(stream, position):
    stream.seek(0, 2)
    end = stream.tell()
    if end <= position:
        return end, []
    # A finishing worker can print a very large JSON manifest in one burst.
    # The complete bytes remain in the file; inspect only a bounded tail here.
    start = max(position, end - 65536)
    stream.seek(start)
    lines = stream.read(end - start).decode('utf-8', errors='replace').splitlines()
    if start > position and lines:
        lines = lines[1:]
    progress = [line[:350] for line in lines
                if line.startswith(_PROGRESS_PREFIXES)]
    return end, progress[-4:]


def _result_summary(path, *, job, plan, account):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    supplied = value.get('manifest_sha256')
    content = {name: item for name, item in value.items() if name != 'manifest_sha256'}
    checksum = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                        separators=(',', ':')).encode('utf-8')).hexdigest()
    if (supplied != checksum or value.get('schema') != 'new-data-kaggle-result-v1'
            or value.get('job') != job or value.get('execution_plan') != plan
            or value.get('account_id') != account
            or value.get('state') not in ('ROUND_COMPLETE', 'CHECKPOINTED_PARTIAL')
            or not isinstance(value.get('tasks'), list)
            or not isinstance(value.get('blocks'), list)):
        raise ValueError('Saved result belongs to another job/plan/account or changed; retain its Outputs.')
    return {'state': value['state'], 'account_id': account,
            'completed_tasks': len(value['tasks']), 'vector_blocks': len(value['blocks']),
            'job': job, 'execution_plan': plan, 'manifest_sha256': supplied}


def run_with_bounded_output(command, *, env, log_path, result_path, job, plan,
                            account, report=print, poll_seconds=2,
                            heartbeat_seconds=30):
    """Run the immutable worker and retain partial artifacts even on failure.

    A valid partial manifest permits a normal notebook finish so Kaggle can save
    its files. A failed worker is explicitly reported as FAILED, never COMPLETE.
    CPU import still verifies every task, producer and vector before reuse.
    """
    log_path, result_path = Path(log_path), Path(result_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_heartbeat, last_progress = started, 'starting locked GPU worker'
    with log_path.open('ab', buffering=0) as output:
        output.write(b'\n--- KAGGLE GPU RUN ---\n')
        with log_path.open('rb') as reader:
            position = log_path.stat().st_size
            process = subprocess.Popen(command, env=env, stdout=output,
                                       stderr=subprocess.STDOUT)
            try:
                while True:
                    position, messages = _read_progress(reader, position)
                    for message in messages:
                        report(message)
                        last_progress = message
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_seconds:
                        report(f'KAGGLE heartbeat | {now-started:.0f}s | {last_progress}'
                               f' | full log: {log_path.name}')
                        last_heartbeat = now
                    if process.poll() is not None:
                        # Capture progress flushed between the read and exit.
                        _, final_messages = _read_progress(reader, position)
                        for message in final_messages:
                            report(message)
                        break
                    time.sleep(poll_seconds)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
    returncode = process.returncode
    if returncode:
        report(f'GPU_RUN_FAILED | return code {returncode} | full traceback: {log_path}')
    if not result_path.is_file():
        raise RuntimeError(f'GPU worker exited {returncode} without a result manifest. '
                           f'Retain Outputs and inspect {log_path}; do not restart the old plan blindly.')
    summary = _result_summary(result_path, job=job, plan=plan, account=account)
    summary.update(process_state='COMPLETE' if returncode == 0 else 'FAILED',
                   returncode=returncode, log_path=str(log_path), result_path=str(result_path))
    report('GPU_RUN_SUMMARY: ' + json.dumps(summary, ensure_ascii=False))
    if returncode:
        report('PARTIAL_OUTPUTS_READY_FOR_SAVE: worker failed; completed tasks remain available '
               'for verified CPU import. This is not full embedding completion.')
    return summary
