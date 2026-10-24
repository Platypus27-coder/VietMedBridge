"""Fetch pinned notebook Outputs without enumerating every output-file page."""
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import os
from pathlib import Path, PurePosixPath
import time

from vietmedbridge.artifacts import atomic_json, digest_json


def _request():
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    return ApiGetKernelRequest()


def retry_kaggle(operation, *, report=print, attempts=6, sleep=time.sleep):
    """Retry only throttling/transient server errors; never retry permission errors."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            if status not in (429, 500, 502, 503, 504) or attempt == attempts-1:
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
            report(f'KAGGLE_RETRY HTTP {status} | same pinned resource | wait {delay:.0f}s '
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

    def _download(self, *, path=None, force=False):
        import kagglehub
        def transfer():
            with _cache_at(self.cache):
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
        self._saved('FETCHING_BUNDLE', manifest_sha256=manifest['manifest_sha256'])
        self.report('KAGGLE_OUTPUT_BUNDLE: download/reuse pinned archive; no output-page enumeration')
        directory = self._download()
        target = directory.joinpath(*PurePosixPath(relative).parts)
        root = target.parent.resolve()
        required = ['job.json', 'new-data-result.json']
        if manifest.get('execution_plan'):
            required.append('execution-plan.json')
        required.extend(block['path'] for block in manifest['blocks'])
        missing = []
        for name in required:
            path = (root/name).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Saved vector path escapes the result directory.')
            if not path.is_file():
                missing.append(name)
        if missing:
            self._saved('BUNDLE_INCOMPLETE', missing_files=len(missing))
            raise RuntimeError(f'Kaggle output archive is incomplete: {len(missing)} required files missing. '
                               'No checkpoint import has started; retain the saved version.')
        self._saved('BUNDLE_READY_FOR_NATIVE_VALIDATION', manifest_sha256=manifest['manifest_sha256'])
        return target
