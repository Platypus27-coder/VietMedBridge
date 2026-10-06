"""Sample pipeline process-tree RSS without an extra runtime dependency."""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

from src.storage.io import atomic_json
from src.utils.environment import environment_info


def process_tree_rss(root_pid: int) -> int | None:
    if sys.platform == 'win32':
        from ctypes import wintypes as w

        class ProcessEntry(ctypes.Structure):
            _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('pid', w.DWORD),
                        ('heap', ctypes.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
                        ('parent', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD), ('exe', w.WCHAR * 260)]

        class MemoryCounters(ctypes.Structure):
            _fields_ = [('cb', w.DWORD), ('page_faults', w.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                                    'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
        kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(ProcessEntry)]
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(MemoryCounters), w.DWORD]
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return None
        parents = {}
        try:
            entry = ProcessEntry()
            entry.size = ctypes.sizeof(entry)
            more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while more:
                parents[entry.pid] = entry.parent
                more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel.CloseHandle(snapshot)
        members = {root_pid}
        while True:
            expanded = members | {pid for pid, parent in parents.items() if parent in members}
            if expanded == members:
                break
            members = expanded
        total = 0
        measured = False
        for pid in members:
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                continue
            try:
                counters = MemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    total += counters.rss
                    measured = True
            finally:
                kernel.CloseHandle(handle)
        return total if measured else None
    parents, rss = {}, {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            tail = (path/'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(path.name)] = int(tail[1])
            rss[int(path.name)] = int((path/'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            continue
    members = {root_pid}
    while True:
        expanded = members | {pid for pid, parent in parents.items() if parent in members}
        if expanded == members:
            break
        members = expanded
    return sum(rss.get(pid, 0) for pid in members) if root_pid in rss else None


class PilotMonitor:
    def __init__(self, root: Path, interval=5.0):
        self.root = root
        self.interval = interval
        self.started = time.monotonic()
        self.stage = 'preflight'
        self.stop = threading.Event()
        self.observations = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        info = environment_info(self.root)
        self.observations.append({'elapsed_seconds': time.monotonic()-self.started, 'stage': self.stage,
                                  'process_tree_rss_bytes': process_tree_rss(os.getpid()),
                                  'ram_available_bytes': info['ram_available_bytes'],
                                  'disk_available_bytes': info['disk_available_bytes']})
        atomic_json(self.root/'checkpoints/pilot_resources.json', self.summary())

    def _run(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def summary(self):
        rss = [r['process_tree_rss_bytes'] for r in self.observations if r['process_tree_rss_bytes'] is not None]
        ram = [r['ram_available_bytes'] for r in self.observations if r['ram_available_bytes'] is not None]
        return {'elapsed_seconds': time.monotonic()-self.started, 'sampling_interval_seconds': self.interval,
                'peak_observed_process_tree_rss_bytes': max(rss) if rss else None,
                'min_observed_ram_available_bytes': min(ram) if ram else None,
                'rss_note': 'Sum of sampled working sets; shared pages may be counted twice and brief peaks may be missed.',
                'observations': self.observations}

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=self.interval+2)
        self._sample()


def storage_measurement(root: Path) -> dict:
    result = {}
    for name in ('data', 'checkpoints', 'crawl_jobs', 'logs', 'reports'):
        files = [p for p in (root/name).rglob('*') if p.is_file()]
        result[name] = {'files': len(files), 'bytes': sum(p.stat().st_size for p in files)}
    return result
