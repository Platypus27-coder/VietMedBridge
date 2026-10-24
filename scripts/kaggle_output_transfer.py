"""Fetch pinned notebook Outputs without enumerating every output-file page."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from email.utils import parsedate_to_datetime
import os
from pathlib import Path, PurePosixPath
import time

from requests.exceptions import ChunkedEncodingError, ConnectionError, SSLError, Timeout

from vietmedbridge.artifacts import atomic_json, digest_json


def _request():
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    return ApiGetKernelRequest()


def retry_kaggle(operation, *, report=print, attempts=6, sleep=time.sleep):
    """Retry a pinned transfer after throttling, server or interrupted-stream errors.

    The Hub downloader retains the partial file. Calling the same operation with
    the same cache lets it use HTTP Range when the server advertises support.
    Do not remove partial payloads or mark them complete here.
    """
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            interrupted = (status is None and isinstance(error, (ChunkedEncodingError, ConnectionError, Timeout))
                           and not isinstance(error, SSLError))
            if (not interrupted and status not in (429, 500, 502, 503, 504)) or attempt == attempts-1:
                raise
            value = (getattr(response, 'headers', {}) or {}).get('Retry-After')
            delay = min(15 * 2**attempt, 120)
            if value:
                try:
                    delay = max(1, float(value))
                except (TypeError, ValueError):
                    try:
                        delay = max(1, parsedate_to_datetime(value).timestamp()-time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
            if delay > 300:
                raise RuntimeError('Kaggle requests a cooldown longer than five minutes; '
                                   'retain this runtime/cache and retry after the server cooldown.') from error
            reason = f'network {type(error).__name__}' if interrupted else f'HTTP {status}'
            report(f'KAGGLE_RETRY {reason} | same pinned resource; partial cache retained | wait {delay:.0f}s '
                   f'| attempt {attempt+1}/{attempts-1}')
            while delay > 0:
                step = min(30, delay)
                sleep(step)
                delay -= step
                if delay > 0:
                    report(f'KAGGLE_COOLDOWN: {delay:.0f}s remaining; no new API request yet')


@contextmanager
def _cache_at(path):
    """Keep large temporary downloads on Colab, without copying a second payload."""
    name = 'KAGGLEHUB_CACHE'
    prior = os.environ.get(name)
    os.environ[name] = str(path)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior


class SavedNotebookOutputs:
    def __init__(self, api, handle, work, *, project=None, report=print):
        self.api, self.handle, self.report = api, handle, report
        owner, slug, *version = handle.split('/')
        if version:
            number = int(version[0])
        else:
            def metadata():
                request = _request()
                request.user_name, request.kernel_slug = owner, slug
                with api.build_kaggle_client() as client:
                    return client.kernels.kernels_api_client.get_kernel(request)
            response = retry_kaggle(metadata, report=report)
            number = response.metadata.current_version_number
        if type(number) is not int or number <= 0:
            raise ValueError('Kaggle did not return a valid saved notebook version; no output download started.')
        self.pinned = f'{owner}/{slug}/versions/{number}'
        self.cache = Path(work)/'kagglehub-output-cache'
        root = Path(project)/'data_staging' if project is not None else Path(work)
        self.record = root/'kaggle_output_downloads'/(digest_json(self.pinned)[:20]+'.json')
        self._saved('VERSION_PINNED')
        report(f'KAGGLE_OUTPUT_VERSION: {self.pinned} | direct manifest, no output-page scan')

    def _saved(self, state, **details):
        atomic_json(self.record, {'handle': self.handle, 'pinned_handle': self.pinned,
                                 'state': state, **details})

    def _download(self, *, path=None, force=False, cache_active=False):
        import kagglehub
        def transfer():
            with (nullcontext() if cache_active else _cache_at(self.cache)):
                # The public Hub resolver downloads a single named file, or the
                # full archived bundle. It does not list output pages for these.
                return Path(kagglehub.notebook_output_download(self.pinned, path=path,
                                                               force_download=force))
        return retry_kaggle(transfer, report=self.report)

    def manifest(self, relative):
        name = PurePosixPath(relative)
        if (not relative or name.is_absolute() or '..' in name.parts or '\\' in relative
                or ':' in relative or name.as_posix() != relative):
            raise ValueError('Unsafe saved manifest path.')
        self._saved('FETCHING_MANIFEST', manifest_path=relative)
        path = self._download(path=relative, force=True)
        if not path.is_file():
            raise RuntimeError('Kaggle did not return the saved result manifest; keep the failed version.')
        self._saved('MANIFEST_DOWNLOADED', manifest_path=relative)
        return path

    def all(self, relative, manifest):
        # A large dynamically generated archive can repeatedly lose its stream
        # without advertising Range support. The manifest already names every
        # asset needed by native install, so neither an archive nor a listing
        # request is necessary. Completed named files retain Hub cache markers.
        self._saved('FETCHING_NAMED_FILES', manifest_sha256=manifest['manifest_sha256'])
        target = self._download(path=relative)
        root = target.parent.resolve()
        required = ['job.json']
        if manifest.get('execution_plan'):
            required.append('execution-plan.json')
        required.extend(block['path'] for block in manifest['blocks'])
        required = list(dict.fromkeys(required))
        # Validate every path before starting additional remote requests.
        for name in required:
            pure = PurePosixPath(name)
            if (not name or pure.is_absolute() or '..' in pure.parts or '\\' in name
                    or ':' in name or pure.as_posix() != name):
                raise ValueError('Unsafe saved vector path.')
            path = (root/name).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Saved vector path escapes the result directory.')
        remote_root = PurePosixPath(relative).parent
        finished = 0
        total = len(required)
        def fetch(name):
            local = self._download(path=(remote_root/name).as_posix(), cache_active=True)
            if not local.is_file() or local.resolve() != (root/name).resolve():
                raise RuntimeError(f'Kaggle did not return required output file: {name}')
            return name
        self.report(f'KAGGLE_OUTPUT_FILES: 0/{total:,} | two named transfers; no archive or output-page scan')
        # Set the process-wide cache once, outside the pool. Per-thread changes
        # of KAGGLEHUB_CACHE could restore each other's environment mid-transfer.
        try:
            with _cache_at(self.cache), ThreadPoolExecutor(max_workers=2) as pool:
                names = iter(required)
                pending = {pool.submit(fetch, name) for name in [next(names, None), next(names, None)]
                           if name is not None}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        finished += 1
                        if finished % 32 == 0 or finished == total:
                            self.report(f'KAGGLE_OUTPUT_FILES: {finished:,}/{total:,} ready in cache '
                                        '| native checksums checked during import')
                        if finished % 128 == 0:
                            self._saved('FETCHING_NAMED_FILES', completed_files=finished, total_files=total,
                                        manifest_sha256=manifest['manifest_sha256'])
                    for _ in done:
                        name = next(names, None)
                        if name is not None:
                            pending.add(pool.submit(fetch, name))
        except Exception:
            self._saved('NAMED_FILES_INTERRUPTED', completed_files=finished, total_files=total,
                        manifest_sha256=manifest['manifest_sha256'])
            raise
        self._saved('BUNDLE_READY_FOR_NATIVE_VALIDATION', manifest_sha256=manifest['manifest_sha256'])
        return target
