"""Use the isolated Kaggle CLI installation; never print authentication material."""
from pathlib import Path
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.tools/kaggle_cli'))
# Kaggle upload-cache filenames escape the native separator only. Normalize
# Windows CLI folder arguments before the upstream client builds these names.
for index, value in enumerate(sys.argv[:-1]):
    if value in ('-p', '--path'):
        sys.argv[index+1] = str(Path(sys.argv[index+1]))
from kaggle.cli import main

if __name__ == '__main__':
    main()
