"""Frozen-input and embedding-map contracts for the isolated ten-source batch.

Native model code and content-cache keys stay unchanged. Only new input parts
are opened. A worker journal replaces repeated scans of old map payloads.
"""
from pathlib import Path


from vietmedbridge.artifacts import (atomic_json, digest_json, publish_file,
    read_json, sha256_file, utc_now, verify_file)
from vietmedbridge.content_embeddings import checked, key, seal
from vietmedbridge.portable_embeddings import resources_relative, validate_encoder, validate_runtime_lock
from vietmedbridge.scale_vectors import bound

try:
    from .new_data_cache_lookup import ResumableContentBlocks
except ImportError:
    from new_data_cache_lookup import ResumableContentBlocks

PARENT = "btc-new-71be0e12-20261019-v1"
FAMILIES = ("bge", "qwen")


def load_new_batch(project):
    project = Path(project).resolve()
    root = project / "data_staging" / PARENT / "freeze_batch_v1"
    batch, contract = checked(read_json(root / "freeze_batch.json")), checked(read_json(root / "contract.json"))
    if (batch.get("state") != "NEW_DATA_INPUTS_READY" or batch.get("completed_builds") != 10
            or batch.get("total_builds") != 10 or batch["contract_sha256"] != contract["manifest_sha256"]
            or len(batch["builds"]) != 10 or len(contract["sources"]) != 10):
        raise ValueError("The complete ten-source frozen input handoff is required.")
    expected = {f"new-{i:05d}" for i in range(9)} | {"retained-71be0e12"}
    if {b["build_run"] for b in batch["builds"]} != expected:
        raise ValueError("New-data batch source coverage changed.")
    sources, ordinal = [], 0
    for item, source in zip(batch["builds"], contract["sources"]):
        checked(item)
        if (item["state"] != "COMPLETE" or item["contract_sha256"] != contract["manifest_sha256"]
                or any(item[k] != source[k] for k in ("build_run", "build_dir", "snapshot_sha256", "counts"))):
            raise ValueError("Frozen source receipt differs from the batch contract.")
        build = bound(project, item["build_dir"])
        inputs = checked(read_json(build / "index_inputs/units.json"))
        if (inputs.get("state") != "COMPLETE" or inputs["manifest_sha256"] != item["index_inputs_manifest_sha256"]
                or inputs["candidate_manifest_sha256"] != item["candidate_manifest_sha256"]
                or inputs["input_count"] != item["unique_dense_inputs"] or len(inputs["parts"]) != item["input_parts"]):
            raise ValueError("Frozen input manifest binding changed.")
        cursor = 0
        for part in inputs["parts"]:
            if (part["start"] != cursor or part["rows"] != min(4096, inputs["input_count"]-cursor)
                    or part["signature"] != inputs["signature"] or not part["path"].endswith(".parquet")):
                raise ValueError("Frozen input part range changed.")
            bound(build / "index_inputs", part["path"])
            sources.append({"ordinal": ordinal, "build": item, "inputs": inputs, "part": part})
            cursor += part["rows"]
            ordinal += 1
        if cursor != inputs["input_count"]:
            raise ValueError("Frozen input coverage changed.")
    if len(sources) != batch["input_parts"]:
        raise ValueError("Batch part coverage changed.")
    return root, batch, sources


def embedding_contract(data_root, checkout):
    """Metadata-only: bind exactly to the existing paid BGE/Qwen producers."""
    data_root, checkout = Path(data_root), Path(checkout)
    runtime_lock = checked(read_json(data_root / "retrieval_embedding_runtime_lock.json"))
    runtime = runtime_lock["runtime"]
    validate_runtime_lock(runtime)
    config = read_json(checkout / "configs/retrieval_full.json")
    active = read_json(data_root / "active_data_candidate.json")
    base = data_root / resources_relative(active, config)
    for name, checksum in runtime_lock["model_code"].items():
        if name not in ("retrieval_models.py", "qwen_models.py") or sha256_file(checkout / "src/vietmedbridge" / name) != checksum:
            raise ValueError("Pinned model producer differs from the existing vector cache.")
    job = {"config": config, "runtime": runtime, "model_code": runtime_lock["model_code"]}
    identities = {}
    for family in FAMILIES:
        saved = read_json(base / family / "config.json")
        validate_encoder(job, family, saved["encoder"], saved["dimension"])
        if saved.get("format") != "content-reference-v1":
            raise ValueError("Existing embedding producer format changed.")
        encoder = saved["encoder"]
        count = encoder.get("parameters_before_quantization", encoder.get("parameters"))
        if type(count) is not int or not 0 < count <= 15_000_000_000:
            raise ValueError("Embedding model parameter budget is not verified.")
        identities[family] = {"encoder": encoder, "dimension": saved["dimension"]}
    return config, runtime, identities, active["candidate_manifest_sha256"]


class LazyEncoder:
    """No weights until a cache lookup finds at least one missing text."""
    def __init__(self, expected, factory):
        self.identity, self.dimension = expected["encoder"], expected["dimension"]
        self.factory, self.owner, self.encoder = factory, None, None

    def encode(self, texts, *, batch_size):
        if self.encoder is None:
            self.owner = self.factory()
            self.encoder = self.owner.for_role("corpus") if hasattr(self.owner, "for_role") else self.owner
            if self.encoder.identity != self.identity or self.encoder.dimension != self.dimension:
                self.close()
                raise ValueError("GPU model identity differs from the paid content cache; no new cache namespace created.")
            from vietmedbridge.qwen_execution import configure_embedding_execution
            configure_embedding_execution(self.encoder)
        return self.encoder.encode(texts, batch_size=batch_size)

    def close(self):
        if self.owner is not None:
            self.owner.close()
            self.owner = self.encoder = None


class NewDataEmbeddings:
    def __init__(self, project, work_dir, identities, *, team_size=1, worker_id=0, batch_loader=None):
        if type(team_size) is not int or team_size < 1 or type(worker_id) is not int or not 0 <= worker_id < team_size:
            raise ValueError("Invalid GPU team/worker assignment.")
        self.project, self.work = Path(project).resolve(), Path(work_dir).resolve()
        self.root, self.batch, self.sources = (batch_loader or load_new_batch)(self.project)
        self.data = self.project / "data"
        self.output = self.root / "embeddings_v1"
        self.identities, self.team_size, self.worker_id = identities, team_size, worker_id
        self.work.mkdir(parents=True, exist_ok=True)
        self.phase = "manifest preflight"
        binding = {"schema_version": 1, "batch": self.batch["manifest_sha256"], "models": identities,
                   "format": "NEW_DATA_CONTENT_EMBEDDINGS_V1"}
        path = self.output / "contract.json"
        if path.is_file() and checked(read_json(path)) != seal(binding):
            raise ValueError("New-data embedding contract changed; retain its checkpoints.")
        atomic_json(path, seal(binding))
        self.binding = digest_json(binding)

    def _key(self, source):
        return source["build"]["build_run"] + ":" + str(source["part"]["start"])

    def _view(self, family, source):
        inputs = source["inputs"]
        expected = self.identities[family]
        identity = {"input_manifest_sha256": inputs["manifest_sha256"], "input_count": inputs["input_count"],
                    "encoder": expected["encoder"], "dimension": expected["dimension"], "format": "content-reference-v1"}
        return self.output / family / source["build"]["build_run"], identity

    def _valid_part(self, family, source, saved):
        checked(saved)
        _, identity = self._view(family, source)
        part = source["part"]
        expected = {"signature": digest_json(identity), "start": part["start"], "rows": part["rows"],
                    "input_sha256": part["sha256"], "path": f"map-{part['start']:012d}.parquet"}
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError("Embedding part receipt differs from frozen input/model policy.")
        return saved

    def _journals(self, family):
        expected = {self._key(s): s for s in self.sources}
        merged = {}
        directory = self.output / family / "progress"
        for path in sorted(directory.glob("worker-*.json")):
            saved = checked(read_json(path))
            if saved.get("binding") != self.binding or saved.get("family") != family:
                raise ValueError("GPU worker journal belongs to another batch/model.")
            for name, record in saved["completed"].items():
                if name not in expected:
                    raise ValueError("GPU journal contains an unknown frozen input part.")
                self._valid_part(family, expected[name], record)
                if name in merged and merged[name] != record:
                    raise ValueError("GPU workers published conflicting immutable map receipts.")
                merged[name] = record
        return merged

    def _save(self, family, completed):
        payload = {"binding": self.binding, "family": family, "team_size": self.team_size,
                   "worker_id": self.worker_id, "completed": completed, "updated_at": utc_now()}
        atomic_json(self.output / family / "progress" / f"worker-{self.worker_id}-of-{self.team_size}.json", seal(payload))

    def _seal_views(self, family, completed):
        grouped = {}
        for source in self.sources:
            grouped.setdefault(source["build"]["build_run"], []).append(source)
        for sources in grouped.values():
            root, identity = self._view(family, sources[0])
            if all(self._key(s) in completed for s in sources):
                path = root / "embeddings.json"
                value = seal(identity | {"signature": digest_json(identity), "state": "COMPLETE",
                    "parts": [completed[self._key(s)] for s in sources]})
                if path.is_file() and checked(read_json(path)) != value:
                    raise ValueError("Completed embedding view changed.")
                if not path.is_file():
                    atomic_json(path, value)
