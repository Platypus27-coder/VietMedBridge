"""Execution-only GPU batches and durable per-stage wall times."""
from contextlib import contextmanager
import time

from .artifacts import atomic_json, utc_now


def inference_batches(total_memory_bytes=None):
    """Start conservatively; model OOM backoff already halves unsafe batches."""
    if total_memory_bytes is None:
        import torch
        if not torch.cuda.is_available():
            return {"embedding": 2, "reranker": 2, "device": "cpu-test"}
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gib = total_memory_bytes / 2**30
    batch = 16 if gib >= 38 else 8 if gib >= 20 else 4
    return {"embedding": batch, "reranker": batch, "gpu_memory_gib": round(gib, 2),
            "oom_policy": "halve-and-retry", "scope": "execution-only; model/input policy unchanged"}


class RuntimeProfile:
    def __init__(self, path):
        self.path = path
        self.started = time.perf_counter()
        self.report = {"started_at": utc_now(), "stages": [], "state": "RUNNING"}

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        row = {"stage": name, "state": "RUNNING"}
        self.report["stages"].append(row)
        self._save()
        print(f"Stage: {name}", flush=True)
        try:
            yield row
        except BaseException:
            row["state"] = "INTERRUPTED"
            raise
        else:
            row["state"] = "COMPLETE"
        finally:
            row["seconds"] = round(time.perf_counter() - started, 3)
            self._save()
            print(f"Stage {name}: {row['seconds']:.1f}s", flush=True)

    def finish(self, state):
        self.report["state"] = state
        self._save()

    def record(self, name, started, **details):
        seconds = round(time.perf_counter() - started, 3)
        self.report["stages"].append({"stage": name, "state": "COMPLETE", "seconds": seconds, **details})
        self._save()
        print(f"Stage {name}: {seconds:.1f}s", flush=True)

    def _save(self):
        self.report["seconds_this_call"] = round(time.perf_counter() - self.started, 3)
        atomic_json(self.path, self.report)
