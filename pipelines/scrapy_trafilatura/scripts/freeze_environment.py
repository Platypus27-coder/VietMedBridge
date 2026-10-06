"""Capture the tested environment with markers for Windows-only packages."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True, check=True)
    lines = [line + '; sys_platform == "win32"' if line.lower().startswith(('pywin32==', 'twisted-iocpsupport=='))
             else line for line in result.stdout.splitlines()]
    destination = Path(__file__).resolve().parents[1] / 'requirements-lock.txt'
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Frozen {len(lines)} dependencies to {destination.name}')
