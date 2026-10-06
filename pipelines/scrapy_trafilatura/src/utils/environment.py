from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path


def is_kaggle() -> bool:
    return Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_writable_root() -> Path:
    return Path('/kaggle/working/vibiomir') if is_kaggle() else get_project_root()


def locate_dataset(name: str, input_root: Path = Path('/kaggle/input')) -> Path:
    candidates = sorted(input_root.rglob(name))
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one {name}; found {len(candidates)}: {candidates}')
    return candidates[0]


def environment_info(root: Path | None = None) -> dict:
    root = root or get_writable_root()
    root.mkdir(parents=True, exist_ok=True)
    ram_available = None
    try:
        import psutil
        ram_available = psutil.virtual_memory().available
    except ImportError:
        if sys.platform == 'win32':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ('total_phys', 'available_phys', 'total_page',
                    'available_page', 'total_virtual', 'available_virtual', 'available_extended')]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError('Unable to read Windows RAM availability.')
            ram_available = status.available_phys
        elif Path('/proc/meminfo').exists():
            lines = Path('/proc/meminfo').read_text().splitlines()
            ram_available = next(int(line.split()[1]) * 1024 for line in lines if line.startswith('MemAvailable:'))
    return {'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'ram_available_bytes': ram_available, 'disk_available_bytes': shutil.disk_usage(root).free,
            'working_directory': str(Path.cwd()), 'writable_root': str(root), 'kaggle': is_kaggle()}
