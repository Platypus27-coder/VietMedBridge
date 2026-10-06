from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from src.crawler.raw_store import RawStore
from src.preprocessing.cleaner import clean_text, content_hash
from src.preprocessing.language import detect_language
from src.preprocessing.quality import assess_quality
from src.storage.io import atomic_json, parquet_rows
from src.storage.manifests import EventWriter, compact_events
from src.storage.schemas import EXTRACT_SCHEMA
from src.storage.index import ManifestIndex
from src.utils.config import output_path

LOGGER = logging.getLogger(__name__)


def extract_one(row: dict, config: dict) -> dict:
    mime = row['effective_content_type']
    kind = 'pdf' if mime == 'application/pdf' else 'html' if mime in ('text/html', 'application/xhtml+xml') else 'text'
    extractor = {'html': 'trafilatura', 'pdf': 'pymupdf', 'text': 'charset-normalizer'}[kind]
    result = {'crawl_url_id': row['crawl_url_id'], 'raw_sha256': row['raw_sha256'], 'snapshot_id': row['snapshot_id'],
              'extractor': extractor, 'extractor_version': version(extractor), 'extract_status': 'EXTRACT_SUCCESS'}
    started = time.monotonic()
    try:
        body = RawStore(config).read(row, config['extraction'][kind]['max_input_bytes'])
        if kind == 'html':
            from .html import extract_html
            extracted = extract_html(body, row['final_url'], config, row.get('declared_charset'), row.get('fetch_url'))
            if extracted is None:
                result['extract_status'] = 'EMPTY_MAIN_TEXT'
                extracted = {'text': '', 'text_markdown': '', 'title': None, 'text_format': 'markdown', 'structure_source': 'html_main_body'}
        elif kind == 'pdf':
            from .pdf import extract_pdf
            extracted = extract_pdf(body, config)
        else:
            from .text import decode_text
            text, charset, method = decode_text(body, row.get('declared_charset'))
            extracted = {'text': text, 'text_markdown': text, 'title': None, 'text_format': 'plain', 'structure_source': 'plain_text'}
        result.update(extracted)
        result['text'] = clean_text(result['text'])
        result['text_markdown'] = clean_text(result['text_markdown'], markdown=result['text_format'] == 'markdown')
        result.update(detect_language(result['text'], config))
        result.update(assess_quality(result['text'], result['text_markdown'], result['title'], config))
        if result.get('source_issue'):
            result.update(quality_tier='QUARANTINE', quality_reason=result['source_issue'].lower())
        if not result['text'] and result['extract_status'] == 'EXTRACT_SUCCESS':
            result['extract_status'] = 'EMPTY_MAIN_TEXT'
        if result['extract_status'] == 'LIKELY_SCANNED_PDF':
            result['quality_tier'], result['quality_reason'] = 'QUARANTINE', 'likely_scanned_pdf'
        result['content_hash'] = content_hash(result['text']) if result['text'] else None
        result['hash_normalization_version'] = 'nfc-whitespace-v1'
    except OverflowError as error:
        result.update(extract_status='EXTRACTION_TOO_LARGE', error_type=type(error).__name__, error_message=str(error))
    except Exception as error:
        # Worker boundary: retain explicit failure; supervisor also handles native process death.
        status = 'RAW_CHECKSUM_FAILED' if isinstance(error, (ValueError, OSError)) and 'checksum' in str(error).lower() else {'html': 'TRAFILATURA_FAILED', 'pdf': 'PDF_PARSE_FAILED', 'text': 'TEXT_DECODE_FAILED'}[kind]
        LOGGER.exception('Extraction failed for %s', row['crawl_url_id'])
        result.update(extract_status=status, error_type=type(error).__name__, error_message=str(error))
    result['extraction_elapsed_ms'] = (time.monotonic()-started)*1000
    return result


def worker_loop(connection, config, task_function=extract_one):
    try:
        for _ in range(config['extraction']['max_tasks_per_child']):
            task = connection.recv()
            if task is None:
                return
            row, attempt = task
            connection.send((task_function(row, config), attempt))
    except EOFError:
        return
    finally:
        connection.close()


def supervised_extract(tasks, config: dict, task_function=extract_one):
    context = mp.get_context('spawn')
    slots = []
    def spawn():
        parent, child = context.Pipe()
        process = context.Process(target=worker_loop, args=(child, config, task_function))
        process.start()
        child.close()
        return {'process': process, 'pipe': parent, 'task': None, 'count': 0}
    def stop(slot):
        if slot['process'].is_alive():
            slot['process'].terminate()
        slot['process'].join(timeout=5)
        if slot['process'].is_alive():
            slot['process'].kill()
            slot['process'].join()
        slot['pipe'].close()
    iterator = iter(tasks)
    exhausted = False
    try:
        slots = [spawn() for _ in range(config['extraction']['workers'])]
        while not exhausted or any(slot['task'] is not None for slot in slots):
            for i, slot in enumerate(slots):
                if slot['task'] is None and not exhausted:
                    if not slot['process'].is_alive() or slot['count'] >= config['extraction']['max_tasks_per_child']:
                        stop(slot)
                        slots[i] = slot = spawn()
                    try:
                        row, attempt = next(iterator)
                    except StopIteration:
                        exhausted = True
                    else:
                        try:
                            slot['pipe'].send((row, attempt))
                        except (EOFError, OSError):
                            stop(slot)
                            slots[i] = spawn()
                            yield {'crawl_url_id': row['crawl_url_id'], 'raw_sha256': row['raw_sha256'],
                                   'extract_status': 'WORKER_CRASHED', 'error_message': 'Worker pipe closed while dispatching.'}, attempt
                            continue
                        kind = 'pdf' if row['effective_content_type'] == 'application/pdf' else 'text' if row['effective_content_type'] == 'text/plain' else 'html'
                        slot['task'] = (row, attempt)
                        slot['started'] = time.monotonic()
                        slot['timeout'] = config['extraction'][kind]['timeout_seconds']
                if slot['task'] is not None:
                    row, attempt = slot['task']
                    try:
                        ready = slot['pipe'].poll()
                    except (EOFError, OSError):
                        stop(slot)
                        slot['task'] = None
                        slots[i] = spawn()
                        yield {'crawl_url_id': row['crawl_url_id'], 'raw_sha256': row['raw_sha256'],
                               'extract_status': 'WORKER_CRASHED', 'error_message': 'Worker pipe poll failed.'}, attempt
                        continue
                    if ready:
                        try:
                            result, received_attempt = slot['pipe'].recv()
                        except (EOFError, OSError):
                            result = {'crawl_url_id': row['crawl_url_id'], 'raw_sha256': row['raw_sha256'], 'extract_status': 'WORKER_CRASHED', 'error_message': 'Worker pipe closed.'}
                        slot['task'] = None
                        slot['count'] += 1
                        yield result, attempt
                    elif not slot['process'].is_alive() or time.monotonic()-slot['started'] > slot['timeout']:
                        status = 'WORKER_CRASHED' if not slot['process'].is_alive() else 'EXTRACTION_TIMEOUT'
                        stop(slot)
                        yield {'crawl_url_id': row['crawl_url_id'], 'raw_sha256': row['raw_sha256'], 'extract_status': status,
                               'error_message': 'Supervised worker stopped; raw/mapping retained.'}, attempt
                        slots[i] = spawn()
            time.sleep(0.01)
    finally:
        for slot in slots:
            stop(slot)


def extract_raw(config: dict) -> dict:
    manifest = output_path(config, 'data/manifests/crawl_manifest.parquet')
    # Code changes must invalidate extraction checkpoints as well as config/version changes.
    implementation_files = [Path(__file__), *[Path(__file__).with_name(name) for name in
                             ('html.py', 'encoding.py', 'structure.py', 'pdf.py', 'text.py')],
                            *[Path(__file__).parents[1]/'preprocessing'/name for name in ('quality.py', 'language.py', 'cleaner.py')]]
    implementation_digest = hashlib.sha256(b''.join(p.read_bytes() for p in implementation_files)).hexdigest()
    extraction_config = {'extraction': config['extraction'], 'quality': config['quality'], 'language': config['language'],
                         'implementation_sha256': implementation_digest}
    config_digest = hashlib.sha256(json.dumps(extraction_config, sort_keys=True).encode()).hexdigest()
    identity = hashlib.sha256((config_digest + version('trafilatura') + version('pymupdf')).encode()).hexdigest()
    directory = output_path(config, f'data/manifests/extraction/{identity}')
    target = output_path(config, 'data/manifests/extraction_manifest.parquet')
    compact_events(directory, target, EXTRACT_SCHEMA, attempt='extraction_attempt_no')
    previous = ManifestIndex(output_path(config, 'checkpoints/extraction_index.sqlite'), target)
    budget = config.get('kaggle_batch', {}).get('extraction_time_limit_seconds', 0)
    deadline = time.monotonic() + budget if budget else None
    stopped_by_time_budget = False
    event_budget = config.get('kaggle_batch', {}).get('max_extraction_event_bytes', 0)
    event_bytes = sum(p.stat().st_size for p in output_path(config, 'data/manifests/extraction').rglob('*') if p.is_file())
    stopped_by_output_budget = False
    def tasks():
        nonlocal stopped_by_time_budget, stopped_by_output_budget
        for row in parquet_rows(manifest):
            if row['status'] != 'SUCCESS_RAW':
                continue
            old = previous.get(row['crawl_url_id'])
            if old and old['raw_sha256'] == row['raw_sha256'] and old['extraction_run_id'] == identity:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                stopped_by_time_budget = True
                return  # In-flight workers drain; unsubmitted raw remains resumable.
            if event_budget and event_bytes >= event_budget:
                stopped_by_output_budget = True
                return
            yield row, (old['extraction_attempt_no'] if old else 0) + 1
    writer = EventWriter(directory, config['storage']['jsonl_records_per_part'])
    processed = 0
    try:
        for result, attempt in supervised_extract(tasks(), config):
            result.update(extraction_attempt_no=attempt, event_seq=1,
                          event_id=f'{result["crawl_url_id"]}:{identity}:{attempt}', extraction_run_id=identity,
                          extractor_config_sha256=config_digest, extraction_timestamp=datetime.now(timezone.utc).isoformat())
            writer.append(result)
            event_bytes += len(json.dumps(result, ensure_ascii=False).encode('utf-8')) + 1
            processed += 1
            atomic_json(output_path(config, 'checkpoints/extraction_state.json'), {'run_id': identity, 'committed_this_run': processed})
    finally:
        writer.close()
        previous.close()
    compact_events(directory, target, EXTRACT_SCHEMA, attempt='extraction_attempt_no')
    # One immutable Parquet part is sufficient for a pilot; output schema is shared with manifest.
    from src.storage.io import write_parquet
    write_parquet(output_path(config, 'data/processed/extracted/part-00000.parquet'), parquet_rows(target), EXTRACT_SCHEMA)
    return {'processed_this_run': processed, 'run_id': identity,
            'stopped_by_time_budget': stopped_by_time_budget,
            'stopped_by_output_budget': stopped_by_output_budget}
