"""Inspect saved Kaggle outputs before downloading large vector checkpoints."""
from __future__ import annotations

from pathlib import Path
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


def _downloaded_paths(response, directory):
    # The API has returned both a file list and (file list, next-page token).
    if isinstance(response, tuple) and len(response) == 2:
        response, token = response
        if token:
            raise RuntimeError("Kaggle chưa tải hết các trang Outputs; cập nhật gói kaggle rồi thử lại.")
    if not isinstance(response, list):
        raise RuntimeError("Kaggle API trả danh sách Outputs không hợp lệ.")
    directory = Path(directory).resolve()
    paths = [Path(item).resolve() for item in response]
    if any(not path.is_relative_to(directory) or not path.is_file() for path in paths):
        raise RuntimeError("Kaggle trả file thiếu hoặc nằm ngoài thư mục tải.")
    return paths


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
        returned = api.kernels_output(handle, str(probe), file_pattern=MANIFEST_PATTERN,
                                      force=True, quiet=True)
    except Exception as error:
        raise RuntimeError(
            f"{handle}: không đọc được trạng thái/Outputs. Kiểm tra KAGGLE_JSON, quyền đọc "
            "notebook và số version (không dùng scriptVersionId làm số version)."
        ) from error
    paths = _downloaded_paths(returned, probe)
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
        returned = api.kernels_output(handle, str(download.resolve()), force=False, quiet=False)
        paths = _downloaded_paths(returned, download)
        target = (download / probe["relative_manifest"]).resolve()
        if target not in paths or _manifest(target, handle) != manifest:
            raise RuntimeError(
                f"{handle}: Outputs đổi sau preflight; chưa nhập Drive. "
                "Chọn owner/notebook/số-version để cố định đúng phiên đã lưu."
            )
        outputs.append(target.parent)
    return outputs
