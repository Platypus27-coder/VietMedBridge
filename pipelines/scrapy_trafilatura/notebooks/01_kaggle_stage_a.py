# %% Environment. Copy/unzip the repository to /kaggle/working before opening.
from pathlib import Path
import json
import subprocess
import sys

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT/'src/cli.py').is_file():
    # Mount the source ZIP as a Kaggle Dataset; extract code to writable storage.
    from zipfile import ZipFile
    archives = sorted(Path('/kaggle/input').rglob('vibiomir_source.zip'))
    if len(archives) != 1:
        raise FileNotFoundError('Unzip the repository under the working directory, or mount exactly one vibiomir_source.zip.')
    PROJECT_ROOT = Path('/kaggle/working/vibiomir_repo')
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with ZipFile(archives[0]) as archive:
        for member in archive.namelist():
            if not (PROJECT_ROOT/member).resolve().is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError(f'Unsafe ZIP member: {member}')
        archive.extractall(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.environment import environment_info, locate_dataset
WORKING_ROOT = Path('/kaggle/working/vibiomir')
print(json.dumps(environment_info(WORKING_ROOT), indent=2))

# %% Install only missing/incompatible pinned dependencies, never a venv.
from scripts.kaggle_bootstrap import bootstrap
bootstrap(WORKING_ROOT)

# %% Locate both dataset files. Multiple candidates fail with a full candidate list.
LINKS = locate_dataset('links_corpus.parquet')
QUERY = locate_dataset('query.parquet')
print(LINKS, QUERY)

# %% Prepare writable output and one effective config.
import yaml
from src.utils.config import load_config, prepare_directories
config_path = WORKING_ROOT/'config.yaml'
values = yaml.safe_load((PROJECT_ROOT/'configs/config.yaml').read_text())
values['environment']['output_dir'] = str(WORKING_ROOT)
values['dataset'].update(links_path=str(LINKS), query_path=str(QUERY))
WORKING_ROOT.mkdir(parents=True, exist_ok=True)
config_path.write_text(yaml.safe_dump(values, allow_unicode=True), encoding='utf-8')
config = load_config(config_path)
prepare_directories(config)

# Fail before the expensive inventory when Run All would need unavailable Internet.
import urllib.request
try:
    urllib.request.urlopen(config['environment']['network_probe_url'], timeout=10).close()
except OSError as error:
    raise RuntimeError('Network unavailable. Use pre-downloaded raw dataset or enable allowed Internet access.') from error

# %% Build inventory and reproducible sample (default 100 URLs).
from src.inventory.build_inventory import build_inventory
from src.inventory.build_shards import sample_inventory, build_shards
build_inventory(config)
sample = sample_inventory(config, size=100, seed=config['dataset']['sample_seed'])
shards = build_shards(config, sample)

# %% Crawl in separate processes.
for shard in shards:
    subprocess.run([sys.executable, str(PROJECT_ROOT/'scripts/03_run_crawl.py'), '--config', str(config_path), '--shard', str(shard)], check=True)

# %% Offline extraction + final outputs. Subprocess keeps spawn safe in notebooks.
for script in ('04_extract_raw.py', '05_build_documents.py', '06_report.py', 'validate_ingestion.py'):
    subprocess.run([sys.executable, str(PROJECT_ROOT/'scripts'/script), '--config', str(config_path)], check=True)

# %% Report and manual review. Do not infer full-corpus coverage from this sample.
report = json.loads((WORKING_ROOT/'reports/stage_a/report.json').read_text())
print(json.dumps(report, ensure_ascii=False, indent=2))

# %% Package reports/manifests/config/logs/processed outputs, excluding raw cache.
from scripts.package_artifacts import package_artifacts
print(package_artifacts(config, Path('/kaggle/working/vibiomir_stage_a_outputs.zip')))
print('Save Version or create an output Dataset. ZIP alone cannot restore raw extraction; persist raw shards separately if needed.')
