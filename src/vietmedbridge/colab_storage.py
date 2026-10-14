"""Reconnect a dropped Colab Drive mount and resume a checkpointed stage once."""
from __future__ import annotations

import errno
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import time

from .artifacts import atomic_json, utc_now


class DriveUnavailableError(RuntimeError):
    """The stage cannot continue until its Drive mount is available again."""


def _drive_mountpoint(root):
    # Do not resolve symlinks through a mount that may already be disconnected.
    root = Path(root)
    mount = Path("/content/drive")
    return mount if root.is_relative_to(mount) else None


def _binding(root):
    # exists()/is_file() alone do not establish that the contents are readable.
    return hashlib.sha256((Path(root) / "active_data_candidate.json").read_bytes()).hexdigest()


def _disconnected(error, mount):
    if not isinstance(error, OSError):
        return False
    if error.filename is not None and not Path(error.filename).is_relative_to(mount):
        return False
    return error.errno in {107, errno.ENOTCONN} or "transport endpoint is not connected" in str(error).lower()


def _remount(mount, *, readonly):
    from google.colab import drive
    # A failed FUSE endpoint can survive force_remount if the old mount is
    # still registered. Flush/unmount it first; it is already known to be
    # disconnected, so a failure here must not block the fresh mount.
    try:
        drive.flush_and_unmount()
    except Exception:
        pass
    drive.mount(str(mount), force_remount=True, readonly=readonly)


def run_with_drive_recovery(operation, root, work_dir, *, readonly=False, max_recoveries=1):
    """Preflight before weights; recover only ENOTCONN, never corruption or quota.

    The operation must publish resumable checkpoints and release resources on
    failure. Recovery runs on the calling thread, after the operation's finally
    blocks (including embedding verification workers) exit. Each attempt
    unmounts the failed FUSE mount, remounts it, and verifies the same active
    candidate before restarting. A changed candidate is refused.
    """
    root, work = Path(root), Path(work_dir)
    if type(max_recoveries) is not int or max_recoveries < 0:
        raise ValueError("max_recoveries must be a non-negative integer.")
    mount = _drive_mountpoint(root)
    if mount is None:
        return operation()
    if work.is_relative_to(mount):
        raise ValueError("Drive recovery work_dir must be on Colab local disk.")
    report_path = work / "drive_recovery" / (hashlib.sha256(str(root).encode()).hexdigest()[:16] + ".json")
    expected, recoveries = None, 0

    def report(state, error=None):
        value = {"state":state, "data_root":str(root), "readonly":readonly,
                 "active_candidate_file_sha256":expected, "recoveries":recoveries,
                 "updated_at":utc_now()}
        if error is not None:
            value["error"] = {"type":type(error).__name__, "message":str(error)}
        try:
            atomic_json(report_path, value)
        except OSError:
            pass  # A diagnostic failure must not hide the original storage error.

    def unavailable(error):
        report("DRIVE_UNAVAILABLE", error)
        return DriveUnavailableError(
            f"Drive vẫn mất kết nối sau {recoveries} lần khôi phục; đã dừng stage. "
            "Giữ runtime để giữ model cache. Xác minh Drive đọc được active_data_candidate.json rồi "
            "chạy lại cell xử lý cùng DATA_ROOT/candidate. Checkpoint hợp lệ sẽ được dùng lại; "
            f"không tự xóa hay tạo lại dữ liệu. Diagnostic local: {report_path}")

    while True:
        try:
            current = _binding(root)
            if expected is not None and current != expected:
                report("CANDIDATE_CHANGED_AFTER_REMOUNT")
                raise ValueError("Active candidate changed after Drive remount; refusing to resume on another corpus.")
            expected = current
            if recoveries:
                report("RESUMING_VERIFIED_CANDIDATE")
                print("Drive reconnected; same candidate verified. Resuming from valid checkpoints.", flush=True)
            result = operation()
            if recoveries:
                report("RECOVERED")
            return result
        except OSError as error:
            if not _disconnected(error, mount):
                raise
            report("DRIVE_DISCONNECTED", error)
            if recoveries >= max_recoveries:
                raise unavailable(error) from error
            recoveries += 1
            delay = min(2 ** (recoveries - 1), 8)
            print(f"Drive disconnected (Errno 107). Recovery {recoveries}/{max_recoveries}: "
                  f"unmount, remount, verify candidate, then resume checkpoints (wait {delay}s).", flush=True)
            try:
                time.sleep(delay)
                _remount(mount, readonly=readonly)
            except Exception as remount_error:
                raise unavailable(remount_error) from remount_error


def drive_recoverable(*, root_arg, work_arg):
    """Guard notebook entry points while preserving their public signatures."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def guarded(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            output_root = arguments.arguments.get("output_root")
            readonly_mount = (arguments.arguments.get("execution_mode") == "cached_strong"
                              and (output_root is None or _drive_mountpoint(output_root) is None))
            return run_with_drive_recovery(lambda:function(*args, **kwargs),
                arguments.arguments[root_arg], arguments.arguments[work_arg],
                readonly=readonly_mount,
                max_recoveries=3)
        return guarded
    return decorate
