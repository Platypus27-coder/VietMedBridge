"""Resume cached outputs via paged signed URLs, avoiding one API call per asset."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
import os
import re
import time
from urllib.parse import urlsplit

import requests

from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file


def _page_request():
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    return ApiListKernelSessionOutputRequest()


def _asset(url, destination, checksum):
    """Publish one complete file; keep a resumable partial on network failure."""
    if urlsplit(url).scheme != 'https' or not urlsplit(url).netloc:
        raise ValueError('Kaggle returned an unsafe download URL.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.vmb-part')
    start = partial.stat().st_size if partial.is_file() else 0
    headers = {'Accept-Encoding':'identity'}
    if start:
        headers['Range'] = f'bytes={start}-'
    with requests.get(url, stream=True, timeout=(15,60), headers=headers) as response:
        response.raise_for_status()
        if start and response.status_code == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
            if not match or int(match[1]) != start or int(match[2])+1 != int(match[3]):
                raise ValueError('Invalid Range response; partial output was retained.')
            expected = int(match[3])
            mode = 'ab'
        elif response.status_code == 200:
            # A server that ignores Range must replace only the temporary file.
            start, mode = 0, 'wb'
            length = response.headers.get('Content-Length')
            expected = int(length) if length and length.isdigit() else None
        else:
            raise ValueError('Unexpected download response; no checkpoint published.')
        with partial.open(mode) as stream:
            for block in response.iter_content(1024*1024):
                if block:
                    stream.write(block)
        if expected is not None and partial.stat().st_size != expected:
            raise requests.exceptions.ChunkedEncodingError('Output stream ended before the declared size')
    if checksum and sha256_file(partial) != checksum:
        raise ValueError('Downloaded vector checksum differs; no checkpoint published.')
    os.replace(partial,destination)
    return destination.stat().st_size


def download_paged_outputs(self, relative, manifest):
    from kagglehub.cache import Cache
    from kagglehub.handle import parse_notebook_handle
    try:
        from .kaggle_output_transfer import _cache_at, retry_kaggle
    except ImportError:
        from kaggle_output_transfer import _cache_at, retry_kaggle
    owner,slug,_,version = self.pinned.split('/')
    handle = parse_notebook_handle(self.pinned)
    with _cache_at(self.cache):
        cache = Cache()
        target = cache.load_from_cache(handle,relative)
        target = Path(target) if target else self._download(path=relative)
        root = target.parent.resolve()
        remote_root = PurePosixPath(relative).parent
        required = {'job.json':None}
        if manifest.get('execution_plan'):
            required['execution-plan.json'] = None
        for block in manifest['blocks']:
            name,checksum = block['path'],block['sha256']
            if name in required and required[name] != checksum:
                raise ValueError('Conflicting output paths in manifest.')
            required[name] = checksum
        for name in required:
            pure = PurePosixPath(name)
            if (not name or pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name
                    or pure.as_posix() != name or not (root/name).resolve().is_relative_to(root)):
                raise ValueError('Unsafe saved output path.')
        remote = {(remote_root/name).as_posix():name for name in required}
        ready = {name for rel,name in remote.items() if cache.load_from_cache(handle,rel)}
        if any(Path(cache.get_path(handle,(remote_root/name).as_posix())).resolve() != (root/name).resolve()
               for name in ready):
            raise ValueError('Output cache location differs from the pinned manifest.')
        total,initial = len(required),len(ready)
        pending = set(required)-ready
        started = time.monotonic()
        last_report = started
        cursor_path = Path(self.cache)/'vmb-output-pages'/(digest_json([self.pinned,manifest['manifest_sha256']])[:20]+'.json')
        cursor = read_json(cursor_path) if cursor_path.is_file() else {}
        valid_cursor = (cursor.get('pinned') == self.pinned and cursor.get('manifest') == manifest['manifest_sha256']
                        and set(cursor.get('processed',[])) <= ready)
        token = cursor.get('next_token','') if valid_cursor else ''
        processed = set(cursor.get('processed',[])) if valid_cursor else set()
        pages = int(cursor.get('pages',0)) if valid_cursor else 0
        seen = set()

        def progress(force=False):
            nonlocal last_report
            now = time.monotonic()
            if not force and now-last_report < 15:
                return
            done = len(ready)-initial
            eta = f'{(total-len(ready))*(now-started)/done/60:.1f} min' if done >= 8 else 'measuring'
            self.report(f'KAGGLE_TRANSFER: {len(ready):,}/{total:,} ready | cached at start {initial:,} '
                        f'| missing {len(pending):,} | download ETA {eta}; Drive import follows')
            self._saved('FETCHING_PAGED_FILES',completed_files=len(ready),total_files=total,
                        missing_files=len(pending),cached_at_start=initial,pages=pages,
                        manifest_sha256=manifest['manifest_sha256'])
            last_report = now

        progress(True)
        if pending:
            while True:
                def page():
                    request = _page_request()
                    request.user_name,request.kernel_slug = owner,slug
                    request.page_size,request.page_token,request.version_label = 200,token,'v'+version
                    with self.api.build_kaggle_client() as client:
                        return client.kernels.kernels_api_client.list_kernel_session_output(request)
                response = retry_kaggle(page,report=self.report)
                pages += 1
                batch = {}
                for item in response.files or []:
                    name = remote.get(item.file_name)
                    if name in pending:
                        batch[name] = item.url
                    if name is not None:
                        processed.add(name)
                self.report(f'KAGGLE_URL_PAGE: {pages} | page size 200 | {len(batch)} missing assets on this page')
                with ThreadPoolExecutor(max_workers=4) as pool:
                    assets = iter(batch.items())
                    jobs = {}
                    def enqueue():
                        item = next(assets,None)
                        if item is not None:
                            name,url = item
                            future = pool.submit(retry_kaggle,
                                lambda:_asset(url,root/name,required[name]),report=self.report)
                            jobs[future] = name
                    for _ in range(4):
                        enqueue()
                    while jobs:
                        done,_ = wait(jobs,timeout=15,return_when=FIRST_COMPLETED)
                        for future in done:
                            name = jobs.pop(future)
                            future.result()
                            cache.mark_as_complete(handle,(remote_root/name).as_posix())
                            ready.add(name)
                            pending.remove(name)
                        for _ in done:
                            enqueue()
                        progress()
                following = response.next_page_token or ''
                if following and (not isinstance(following,str) or following == token or following in seen):
                    raise RuntimeError('Kaggle repeated an output-page token; cached files retained.')
                seen.add(token)
                atomic_json(cursor_path,{'pinned':self.pinned,'manifest':manifest['manifest_sha256'],
                    'next_token':following,'pages':pages,'processed':sorted(processed)})
                progress(True)
                if not pending:
                    break
                if not following:
                    self._saved('PAGED_OUTPUTS_MISSING',missing_files=len(pending),total_files=total,
                                manifest_sha256=manifest['manifest_sha256'])
                    raise RuntimeError(f'{len(pending)} required files absent from saved Outputs; no import started.')
                token = following
        self._saved('BUNDLE_READY_FOR_NATIVE_VALIDATION',completed_files=total,total_files=total,
                    manifest_sha256=manifest['manifest_sha256'])
        self.report(f'KAGGLE_TRANSFER_COMPLETE: {total:,}/{total:,}; native validation and Drive import next')
        return target


def install_paged_output_transfer():
    try:
        from . import kaggle_output_transfer as transport
    except ImportError:
        import kaggle_output_transfer as transport
    transport.SavedNotebookOutputs.all = download_paged_outputs


if __name__ == '__main__':
    install_paged_output_transfer()
    print('Đã bật tải theo trang URL, giữ cache. Chạy lại riêng cell 3 trong cùng runtime.',flush=True)
