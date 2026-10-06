"""Detached Windows pilot supervision. All retries retain the same bounded frontier."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage.io import atomic_json
from src.utils.config import load_config, output_path


def launch_supervisor(config, stage_a_root, size):
    from src.pilot import require_pilot_review
    require_pilot_review(config, stage_a_root, size)
    if sys.platform != 'win32':
        raise RuntimeError('Detached supervisor is currently supported on Windows only.')
    state_path = output_path(config, 'checkpoints/supervisor_state.json')
    if state_path.exists():
        existing = json.loads(state_path.read_text())
        if existing.get('status') == 'RUNNING':
            raise RuntimeError('A supervisor is already recorded as running; inspect it before launching another writer.')
    # Read process IDs only; command lines are never printed or persisted.
    folder_pattern = re.escape(Path(config['_output_root']).name).replace("'", "''")
    executable_prefix = (str(Path(config['_project_root']))+'\\').replace("'", "''")
    command = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { "
               " $_.CommandLine -match 'scripts[/\\\\]run_stage_b\\.py' -and $_.CommandLine -notmatch 'background-supervisor' "
               f" -and $_.CommandLine -match '{folder_pattern}' -and $_.ExecutablePath.StartsWith('{executable_prefix}') "
               " } | Select-Object -ExpandProperty ProcessId | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True, text=True, check=True)
    found = json.loads(result.stdout) if result.stdout.strip() else []
    pids = found if isinstance(found, list) else [found]
    live_state_path = output_path(config, 'checkpoints/run_state.json')
    if not pids and live_state_path.exists() and json.loads(live_state_path.read_text()).get('status') == 'RUNNING':
        raise RuntimeError('An active crawl checkpoint exists but its runner PID was not found. Refusing to create a second writer.')
    # Do not supervise a different pilot accidentally.
    if pids:
        authorization = json.loads(output_path(config, 'checkpoints/stage_b_authorization.json').read_text())
        if authorization['max_urls_this_run'] != size or Path(authorization['stage_a_root']) != stage_a_root:
            raise ValueError('Active pilot authorization differs from the supervisor request.')
    frozen = output_path(config, 'checkpoints/supervisor_config.yaml')
    effective = output_path(config, 'checkpoints/effective_config.yaml')
    shutil.copyfile(effective if effective.exists() else Path(config['_config_path']), frozen)
    log = output_path(config, 'logs/supervisor.log')
    arguments = [sys.executable, str(Path(__file__).resolve()), '--config', str(frozen), '--output-dir', config['_output_root'],
                 '--stage-a-root', str(stage_a_root), '--sample-size', str(size), '--watch-pids', json.dumps(pids)]
    with log.open('ab') as handle:
        process = subprocess.Popen(arguments, cwd=config['_project_root'], stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
                                   creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    return {'supervisor_pid': process.pid, 'watch_pids': pids, 'log': str(log), 'max_urls': size, '100k_authorized': False}


def wait_for_existing(pids):
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handles = [kernel.OpenProcess(0x100000 | 0x1000, False, pid) for pid in pids]
    handles = [handle for handle in handles if handle]
    try:
        while any(kernel.WaitForSingleObject(handle, 0) == 258 for handle in handles):
            time.sleep(5)
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)


def supervise(config, stage_a_root, size, pids):
    from src.pilot import require_pilot_review
    require_pilot_review(config, stage_a_root, size)
    state_path = output_path(config, 'checkpoints/supervisor_state.json')
    def state(status, phase, **extra):
        atomic_json(state_path, {'status': status, 'phase': phase, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                'supervisor_pid': __import__('os').getpid(), 'watch_pids': pids, 'max_urls': size, **extra})
    def complete():
        path = output_path(config, 'reports/stage_b/report.json')
        if not path.exists():
            return False
        report = json.loads(path.read_text())
        return report.get('next_stage', {}).get('100k_authorized') is False and report.get('unique_urls') == size
    try:
        state('RUNNING', 'waiting_for_existing_pilot')
        wait_for_existing(pids)
        for attempt in range(2):
            if complete():
                break
            state('RUNNING', 'resume_pilot', recovery_attempt=attempt+1)
            command = [sys.executable, 'scripts/run_stage_b.py', '--config', config['_config_path'], '--output-dir', config['_output_root'],
                       '--stage-a-root', str(stage_a_root), '--sample-size', str(size)]
            subprocess.run(command, cwd=config['_project_root'], check=False)
        if not complete():
            raise RuntimeError('Bounded recovery did not complete; manifests and JOBDIR remain available.')
        state('RUNNING', 'packing_and_verification')
        from scripts.finalize_stage_b import finalize
        finalized = finalize(config)
        state('RUNNING', 'packaging')
        from scripts.package_artifacts import package_artifacts
        from scripts.package_source import package_source
        destination = Path(config['_project_root'])/'outputs/vibiomir_stage_b_10k_outputs.zip'
        package_artifacts(config, destination)
        package_source(Path(config['_project_root'])/'outputs/vibiomir_source.zip')
        state('COMPLETE', 'finished', artifacts_zip=str(destination), packing=finalized['packing'])
        print(json.dumps({'status': 'COMPLETE', 'report': str(output_path(config, 'reports/stage_b/report.json'))}, indent=2), flush=True)
    except BaseException as error:
        state('FAILED', 'inspect_supervisor_log', error_type=type(error).__name__, error_message=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stage-a-root', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, required=True)
    parser.add_argument('--watch-pids', required=True)
    args = parser.parse_args()
    supervise(load_config(args.config, args.output_dir), args.stage_a_root.resolve(), args.sample_size, json.loads(args.watch_pids))
