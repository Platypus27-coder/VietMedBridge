"""Inspect saved Kaggle outputs before downloading large vector checkpoints."""
from __future__ import annotations

import inspect
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from .artifacts import digest_json, read_json


MANIFEST_PATTERN = r"(^|/)result-manifest[.]json$"
HANDLE_PATTERN = r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[0-9]+)?"


def output_handles(value):
    handles = [item.strip() for item in value.split(",")]
    if len(set(handles)) != len(handles) or any(
        not re.fullmatch(HANDLE_PATTERN, handle) for handle in handles
    ):
        raise ValueError("Điền username/notebook-slug khác nhau, cách nhau bởi dấu phẩy; có thể thêm /số-version.")
    return handles


def _downloaded_page(response, directory):
    # The API has returned both a file list and (file list, next-page token).
    token = None
    if isinstance(response, tuple) and len(response) == 2:
        response, token = response
    if not isinstance(response, list):
        raise RuntimeError("Kaggle API trả danh sách Outputs không hợp lệ.")
    directory = Path(directory).resolve()
    paths = [Path(item).resolve() for item in response]
    if any(not path.is_relative_to(directory) or not path.is_file() for path in paths):
        raise RuntimeError("Kaggle trả file thiếu hoặc nằm ngoài thư mục tải.")
    return paths, token


def _sdk_request():
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    return ApiListKernelSessionOutputRequest()


def _stream_file(url, destination, *, force):
    """Finish one download atomically; a network failure preserves its old copy."""
    import requests

    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        identity = response.headers.get("Content-Encoding", "identity") in ("", "identity")
        length = int(length) if identity and length and length.isdigit() else None
        if not force and destination.is_file() and length is not None and destination.stat().st_size == length:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".vmb-", delete=False) as stream:
                temporary = Path(stream.name)
                written = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        written += len(chunk)
            if length is not None and written != length:
                raise RuntimeError("Kaggle download chưa đủ bytes; giữ nguyên file hoàn tất trước đó.")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _sdk_output_page(api, handle, directory, token, *, file_pattern, force, quiet):
    """Kaggle 2.0.2 exposes next_page_token only through its underlying SDK."""
    owner, slug, *version = handle.split("/")
    request = _sdk_request()
    request.user_name, request.kernel_slug = owner, slug
    request.page_token = token
    if version:
        if not hasattr(request, "version_label"):
            raise RuntimeError("Kaggle SDK chưa hỗ trợ chọn số version; cần cập nhật gói kaggle.")
        request.version_label = "v" + version[0]
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    directory = Path(directory).resolve()
    pattern = re.compile(file_pattern) if file_pattern else None
    paths = []
    for item in response.files or []:
        if pattern and not pattern.search(item.file_name):
            continue
        name = item.file_name
        if (not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts or PurePosixPath(name).as_posix() != name):
            raise RuntimeError("Kaggle trả đường dẫn Outputs không an toàn.")
        destination = (directory / name).resolve()
        if not destination.is_relative_to(directory):
            raise RuntimeError("Kaggle trả đường dẫn nằm ngoài thư mục tải.")
        _stream_file(item.url, destination, force=force)
        paths.append(str(destination))
        if not quiet:
            print(f"Output file downloaded/reused: {destination}", flush=True)
    return paths, response.next_page_token


def _download_pages(api, handle, directory, *, file_pattern=None, force=False, quiet=True, report=print):
    """Read every metadata page, including pages with zero matching files."""
    parameters = inspect.signature(api.kernels_output).parameters
    paging = "page_token" in parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()
    )
    options = {"file_pattern": file_pattern, "force": force, "quiet": quiet}
    # Empty token disables newer clients' internal auto-paging so this loop owns it.
    if paging:
        response = api.kernels_output(handle, str(directory), page_token="", **options)
    elif handle.count("/") == 2:
        response = _sdk_output_page(api, handle, directory, None, **options)
    else:
        response = api.kernels_output(handle, str(directory), **options)
    paths, token = _downloaded_page(response, directory)
    seen, pages = set(), 1
    while token:
        if not isinstance(token, str) or token in seen:
            raise RuntimeError(f"{handle}: Kaggle lặp token phân trang; chưa nhập Drive.")
        seen.add(token)
        if pages == 1 or pages % 25 == 0:
            report(f"{handle} | output_pages={pages} | matched_files={len(paths)} | reading_next_page")
        if paging:
            response = api.kernels_output(handle, str(directory), page_token=token, **options)
        else:
            response = _sdk_output_page(api, handle, directory, token, **options)
        page_paths, token = _downloaded_page(response, directory)
        paths.extend(page_paths)
        pages += 1
    report(f"{handle} | output_pages={pages} | matched_files={len(set(paths))} | listing_complete")
    return list(dict.fromkeys(paths))


def _manifest(path, handle):
    try:
        value = read_json(path)
        if (not isinstance(value, dict) or value.get("schema") != "portable-result-v1"
            or not isinstance(value.get("receipts"), list)
            or not re.fullmatch(r"[0-9a-f]{64}", value.get("job", ""))
            or value.get("manifest_sha256") != digest_json({
                key: item for key, item in value.items() if key != "manifest_sha256"
            })):
            raise ValueError("Invalid portable result manifest.")
        return value
    except (ValueError, TypeError, OSError) as error:
        raise RuntimeError(f"{handle}: result-manifest.json không hợp lệ; chưa nhập Drive.") from error


def inspect_kaggle_output(api, handle, work_dir, *, report=print):
    """Download only the remote manifest; stale local files never count as outputs."""
    output_handles(handle)
    base = Path(work_dir) / "output_preflight" / digest_json(handle)[:16]
    base.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix="probe-", dir=base)).resolve()
    try:
        response = api.kernels_status(handle)
        status = response.get("status") if isinstance(response, dict) else response.status
        status = getattr(status, "name", str(status))
        report(f"Kaggle preflight: {handle} | status={status}")
        paths = _download_pages(api, handle, probe, file_pattern=MANIFEST_PATTERN,
                                force=True, quiet=True, report=report)
    except RuntimeError as error:
        raise RuntimeError(f"{handle}: {error}") from error
    except Exception as error:
        raise RuntimeError(
            f"{handle}: không đọc được trạng thái/Outputs. Kiểm tra KAGGLE_JSON, quyền đọc "
            "notebook và số version (không dùng scriptVersionId làm số version)."
        ) from error
    manifests = [path for path in paths if path.name == "result-manifest.json"]
    names = [path.relative_to(probe).as_posix() for path in manifests]
    report(f"{handle} | remote_manifests={len(manifests)} | paths={names}")
    if len(manifests) != 1:
        raise RuntimeError(
            f"{handle}: Outputs trả {len(manifests)} result-manifest.json (status={status}, "
            f"paths={names}). Kiểm tra Outputs của version đã chạy; nếu vừa lưu bản mới "
            "không có checkpoint, dùng owner/notebook/số-version của bản có checkpoint. "
            f"Metadata/log tải tại {probe}. Chưa tải vector hoặc nhập Drive."
        )
    manifest = _manifest(manifests[0], handle)
    assignment = manifest.get("assignment") or {}
    report(f"{handle} | saved_parts={len(manifest['receipts'])} | account_id={assignment.get('account_id')}")
    return {"handle": handle, "status": status, "manifest": manifest,
            "relative_manifest": manifests[0].relative_to(probe).as_posix()}


def download_kaggle_outputs(api, handles, work_dir, *, report=print):
    """Preflight every account before downloading vectors; preserve valid local downloads."""
    probes, errors = [], []
    for handle in handles:
        try:
            probes.append(inspect_kaggle_output(api, handle, work_dir, report=report))
        except RuntimeError as error:
            errors.append(str(error))
            report(str(error))
    if errors:
        raise RuntimeError("Kaggle Outputs chưa sẵn sàng; chưa tải vector hoặc nhập Drive:\n" + "\n".join(errors))
    if not probes:
        raise ValueError("Cần ít nhất một notebook Outputs.")
    if len({item["manifest"]["job"] for item in probes}) != 1:
        raise RuntimeError("Hai Outputs thuộc hai input job khác nhau; chưa tải vector hoặc nhập Drive.")
    accounts = [item["manifest"]["assignment"]["account_id"] for item in probes
                if item["manifest"].get("assignment")]
    if len(set(accounts)) != len(accounts):
        raise RuntimeError("Các Outputs trùng KAGGLE_ACCOUNT_ID; chọn đúng một version cho mỗi account ID.")
    outputs = []
    for probe in probes:
        handle, manifest = probe["handle"], probe["manifest"]
        legacy = Path(work_dir) / "outputs" / digest_json(handle)[:16]
        cached_manifest = legacy / probe["relative_manifest"]
        try:
            reuse = cached_manifest.is_file() and read_json(cached_manifest) == manifest
        except (ValueError, OSError):
            reuse = False
        download = legacy if reuse else legacy / manifest["manifest_sha256"][:16]
        download.mkdir(parents=True, exist_ok=True)
        report(f"Tải/reuse Outputs: {handle} -> {download}")
        # Without force, Kaggle can reuse already-downloaded files. install_result
        # still checks every required checksum before writing any Drive artifact.
        paths = _download_pages(api, handle, download.resolve(), force=False, quiet=False, report=report)
        target = (download / probe["relative_manifest"]).resolve()
        if target not in paths or _manifest(target, handle) != manifest:
            raise RuntimeError(
                f"{handle}: Outputs đổi sau preflight; chưa nhập Drive. "
                "Chọn owner/notebook/số-version để cố định đúng phiên đã lưu."
            )
        outputs.append(target.parent)
    return outputs
