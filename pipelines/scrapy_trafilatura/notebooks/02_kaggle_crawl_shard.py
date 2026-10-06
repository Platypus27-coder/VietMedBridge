# %% Parameters: one shard per saved run, CPU only, Internet ON.
import time
SESSION_STARTED_AT = time.time()
SHARD_ID = 0
# For a PARTIAL shard, mount its previous output tar + sha256 as input.
# Set an explicit path when multiple versions of the same shard are mounted.
RESTORE_BUNDLE = None
AUTO_RESTORE = True
REQUIRE_RESTORE = False  # Resume deployments set True to refuse a fresh crawl.
EXPECTED_RESTORE_SHA256 = None
# One shared deadline includes dependencies, restore and frontier preparation.
# Finish before Kaggle's 12h maximum; reserve extraction and export time.
SESSION_HOURS = 11
EXTRACTION_RESERVE_MINUTES = 60
EXPORT_RESERVE_MINUTES = 30
CRAWL_HOURS = None  # Optional additional cap; None uses the session deadline.
EXTRACTION_HOURS = None
EXTRACT = True
MAX_RAW_GIB = 4
MAX_EXTRACTION_EVENT_GIB = 1
PROGRESS_INTERVAL_SECONDS = 30
PROGRESS_STALE_SECONDS = 300

# %% Load the reviewed pipeline source from the private campaign Dataset.
from pathlib import Path
import json
import subprocess
import sys
from zipfile import ZipFile
import shutil

input_root = Path('/kaggle/input')
code_markers = sorted(input_root.rglob('source_package_manifest.json'))
if len(code_markers) > 1:
    raise ValueError('Attach only one version of the separate runner Dataset.')
code_input_root = code_markers[0].parent if code_markers else input_root
archives = sorted(code_input_root.rglob('vibiomir_source.zip'))
PROJECT_ROOT = Path('/kaggle/working/vibiomir_repo')
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
if len(archives) == 1:
    with ZipFile(archives[0]) as archive:
        for member in archive.namelist():
            if not (PROJECT_ROOT/member).resolve().is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError('Unsafe source ZIP member.')
        archive.extractall(PROJECT_ROOT)
elif not archives:
    # Some Dataset upload flows expand ZIP files before mounting inputs.
    roots = {p.parent.parent for p in code_input_root.rglob('cli.py') if p.parent.name == 'src'}
    if len(roots) != 1:
        raise FileNotFoundError('Attach exactly one campaign Dataset containing the source ZIP or expanded source.')
    source_root = roots.pop()
    for folder in ('src', 'scripts', 'configs', 'tests', 'notebooks'):
        if (source_root/folder).is_dir():
            shutil.copytree(source_root/folder, PROJECT_ROOT/folder, dirs_exist_ok=True)
    for file in source_root.iterdir():
        if file.is_file() and (file.name.startswith('requirements') or file.name in ('pyproject.toml', 'scrapy.cfg', '.python-version')):
            shutil.copyfile(file, PROJECT_ROOT/file.name)
else:
    raise ValueError('Multiple source ZIPs found; attach only the selected campaign.')
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.kaggle_bootstrap import bootstrap
bootstrap(PROJECT_ROOT)

# %% Find the immutable campaign. The original native BTC IDs are in its mapping.
from src.utils.environment import locate_dataset
campaign_path = locate_dataset('campaign_manifest.json')
CAMPAIGN_ROOT = campaign_path.parent
campaign = json.loads(campaign_path.read_text(encoding='utf-8'))
if not 0 <= SHARD_ID < campaign['shard_count']:
    raise ValueError('SHARD_ID is outside this campaign.')
print(json.dumps({k: campaign[k] for k in ('snapshot_id', 'source_rows', 'unique_urls', 'shard_count')}, indent=2))
print('Selected frontier:', campaign['shards'][SHARD_ID])

# %% Configure a bounded session; no full working inventory is rebuilt.
import yaml
from src.utils.session_budget import stage_time_limit
if not 0 < SESSION_HOURS <= 11 or min(EXTRACTION_RESERVE_MINUTES, EXPORT_RESERVE_MINUTES) <= 0:
    raise ValueError('Use a session of at most 11h with positive extraction/export reserves.')
if SESSION_HOURS*60 <= EXPORT_RESERVE_MINUTES + (EXTRACTION_RESERVE_MINUTES if EXTRACT else 0):
    raise ValueError('Session reserves leave no time for crawling.')
values = yaml.safe_load((PROJECT_ROOT/'configs/config.yaml').read_text(encoding='utf-8'))
WORKING_ROOT = Path('/kaggle/working/vibiomir_batch')
values['environment'].update(mode='kaggle', output_dir=str(WORKING_ROOT))
values['environment']['deployment'].update(target='kaggle', provider='kaggle',
    stage_b_max_urls=100000, disk_budget_bytes=14*1024**3, egress_budget_bytes=None)
values['dataset'].update(links_path=str(CAMPAIGN_ROOT/'links_corpus.parquet'),
                         query_path=str(CAMPAIGN_ROOT/'query.parquet'))
values['crawler'].update(concurrent_requests=16, concurrent_requests_per_domain=2,
    session_time_limit_seconds=(CRAWL_HOURS or 0)*3600, reactor_threadpool_maxsize=32)
values['extraction']['workers'] = 2
values['kaggle_batch'] = {'extract': EXTRACT, 'max_raw_bytes': MAX_RAW_GIB*1024**3,
    'max_extraction_event_bytes': MAX_EXTRACTION_EVENT_GIB*1024**3,
    'progress_interval_seconds': PROGRESS_INTERVAL_SECONDS,
    'progress_stale_seconds': PROGRESS_STALE_SECONDS,
    'min_free_disk_bytes': 3*1024**3, 'extraction_time_limit_seconds': (EXTRACTION_HOURS or 0)*3600,
    'session_deadline_unix': SESSION_STARTED_AT + SESSION_HOURS*3600,
    'extraction_reserve_seconds': EXTRACTION_RESERVE_MINUTES*60,
    'export_reserve_seconds': EXPORT_RESERVE_MINUTES*60}
print('Session budget:', json.dumps({
    'session_hours': SESSION_HOURS, 'crawl_fixed_hours': CRAWL_HOURS,
    'extraction_reserve_minutes': EXTRACTION_RESERVE_MINUTES,
    'export_reserve_minutes': EXPORT_RESERVE_MINUTES,
    'crawl_seconds_remaining': round(stage_time_limit(values, 'crawl'), 1)}))
CONFIG_PATH = PROJECT_ROOT/'kaggle_config.yaml'
CONFIG_PATH.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding='utf-8')

# %% Resume only this shard from its saved, checksummed bundle.
from src.utils.restore_input import select_restore
RESTORE_BUNDLE = select_restore(input_root, SHARD_ID, explicit=RESTORE_BUNDLE,
    auto=AUTO_RESTORE, required=REQUIRE_RESTORE, expected_sha256=EXPECTED_RESTORE_SHA256)
print('Restore:', RESTORE_BUNDLE)

# %% Fail early if Internet is unavailable; crawl runs in a separate process.
import urllib.request
try:
    urllib.request.urlopen(values['environment']['network_probe_url'], timeout=15).close()
except OSError as error:
    raise RuntimeError('Enable Internet in Kaggle notebook settings before Save & Run All.') from error
command = [sys.executable, '-u', str(PROJECT_ROOT/'scripts/run_kaggle_shard.py'),
    '--config', str(CONFIG_PATH), '--campaign-root', str(CAMPAIGN_ROOT),
    '--shard-id', str(SHARD_ID), '--compact-output']
if RESTORE_BUNDLE:
    command.extend(['--restore', RESTORE_BUNDLE])
subprocess.run(command, cwd=PROJECT_ROOT, check=True)

# %% Saved output: one tar containing raw/events/checkpoints, SHA256 and report.
report_path = Path(f'/kaggle/working/vibiomir_shard_{SHARD_ID:05d}_report.json')
report = json.loads(report_path.read_text(encoding='utf-8'))
print(json.dumps({k: v for k, v in report.items() if k != 'domains'}, ensure_ascii=False, indent=2))
if report['status'] == 'PARTIAL':
    print('Save this output. Mount its tar + sha256 in the next run and keep the same SHARD_ID.')
    if report['crawler_finish_reason'] == 'raw_storage_budget' or (report.get('extraction_run') or {}).get('stopped_by_output_budget'):
        print('A cumulative storage cap was reached. Check sizes before raising its parameter; an unchanged cap will pause again.')
else:
    print('Shard complete. Save this output, then increment SHARD_ID in the next sequential run.')
print('Global dedup/native ID mapping are finalized after all shards; this report covers only this shard.')
