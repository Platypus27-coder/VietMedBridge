# Run once in the existing Colab runtime, then rerun only cell 3.
import time
import requests
import kaggle_output_transfer as transfer
from pathlib import PurePosixPath

# Re-running this patch must not nest another retry layer.
transfer._before_stream_retry = getattr(transfer, '_before_stream_retry', transfer.retry_kaggle)


def retry_interrupted_download(operation, **options):
    for attempt in range(6):
        try:
            return transfer._before_stream_retry(operation, **options)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
            if isinstance(error, requests.exceptions.SSLError) or attempt == 5:
                raise
            delay = min(5 * 2**attempt, 30)
            print(f'KAGGLE_NETWORK_RETRY {attempt+1}/5 | wait {delay}s | partial cache retained', flush=True)
            time.sleep(delay)


transfer.retry_kaggle = retry_interrupted_download


def download_named_outputs(self, relative, manifest):
    target = self._download(path=relative)
    root = target.parent.resolve()
    names = ['job.json']
    if manifest.get('execution_plan'):
        names.append('execution-plan.json')
    names.extend(block['path'] for block in manifest['blocks'])
    names = list(dict.fromkeys(names))
    for name in names:
        pure = PurePosixPath(name)
        if (not name or pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name
                or pure.as_posix() != name or not (root/name).resolve().is_relative_to(root)):
            raise ValueError('Unsafe saved output path')
    self._saved('FETCHING_NAMED_FILES', manifest_sha256=manifest['manifest_sha256'])
    self.report(f'KAGGLE_OUTPUT_FILES: 0/{len(names):,} | named files only; no archive or page scan')
    for index, name in enumerate(names, 1):
        path = self._download(path=(PurePosixPath(relative).parent/name).as_posix())
        if not path.is_file() or path.resolve() != (root/name).resolve():
            raise RuntimeError(f'Missing saved output: {name}')
        if index % 32 == 0 or index == len(names):
            self.report(f'KAGGLE_OUTPUT_FILES: {index:,}/{len(names):,} ready; native checksum checks on import')
    self._saved('BUNDLE_READY_FOR_NATIVE_VALIDATION', manifest_sha256=manifest['manifest_sha256'])
    return target


transfer.SavedNotebookOutputs.all = download_named_outputs
print('Đã chuyển sang tải từng file cần thiết. Giữ runtime, chạy lại riêng cell 3; không chạy lại GPU.', flush=True)
