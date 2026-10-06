"""Create a portable source ZIP without datasets, environments or outputs."""
import argparse
from pathlib import Path
import zipfile


def package_source(destination: Path):
    root = Path(__file__).resolve().parents[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src', 'scripts', 'configs', 'tests', 'notebooks'):
            for path in sorted((root/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    archive.write(path, str(path.relative_to(root)))
        for name in ('.python-version', '.gitignore', 'requirements.txt', 'requirements-kaggle.txt', 'requirements-lock.txt',
                     'pyproject.toml', 'scrapy.cfg', 'README.md', 'IMPLEMENTATION_STATUS.md', 'PLAN_Scrapy_Trafilatura_ViBioMIR.md'):
            archive.write(root/name, name)
    temporary.replace(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('outputs/vibiomir_source.zip'))
    print(package_source(parser.parse_args().destination))
