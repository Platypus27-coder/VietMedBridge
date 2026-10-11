"""Portable, persistent IVF/SQ8 indexes over existing immutable vector parts.

Each shard has local row IDs. The sealed view binds those IDs to frozen vector
positions, which DiskCatalog resolves to all official document/chunk aliases.
Templates and shards are content-addressed independently of the corpus view:
appending parts reuses earlier shards without encoding or indexing them again.
"""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import shutil

import numpy as np

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, verify_file
from .content_embeddings import seal
from .embeddings import validate_vectors
from .scale_vectors import bound, check_manifest


DEFAULTS = {"backend":"faiss_ivf_sq8", "parts_per_shard":32, "nlist":1024,
    "training_rows":32768, "training_iterations":12, "threads":4,
    "nprobe":64, "max_nprobe":512, "query_batch_size":16,
    "recall_queries":8, "recall_k":20, "minimum_recall":0.95,
    "qdrant_hnsw_ef":256, "qdrant_batch_size":256}


class VectorIndexPending(RuntimeError):
    def __init__(self,report):
        self.report = report
        super().__init__(report["state"])


def policy(options=None):
    options = options or {}
    if set(options)-set(DEFAULTS):
        raise ValueError(f"Unknown vector index options: {sorted(set(options)-set(DEFAULTS))}")
    result = DEFAULTS | options
    if result["backend"] not in ("faiss_ivf_sq8", "qdrant", "exact"):
        raise ValueError("Vector backend must be faiss_ivf_sq8, qdrant or exact.")
    for key,value in result.items():
        if key not in ("backend", "minimum_recall") and (type(value) is not int or value < 1):
            raise ValueError(f"Invalid vector index option: {key}")
    if not 0 < result["minimum_recall"] <= 1 or result["max_nprobe"] < result["nprobe"]:
        raise ValueError("Invalid ANN recall/probe policy.")
    return result


def groups(store, count):
    check_manifest(store.manifest)
    cursor = 0
    for start in range(0,len(store.manifest["parts"]),count):
        parts = store.manifest["parts"][start:start+count]
        for part in parts:
            if part["start"] != cursor or part["rows"] < 1:
                raise ValueError("Vector index source coverage is not contiguous.")
            cursor += part["rows"]
        # Source signatures/absolute paths may change in a cumulative candidate;
        # vector/input checksums and order, not its directory name, define reuse.
        identity = {"encoder":store.manifest["encoder"], "dimension":store.manifest["dimension"],
            "parts":[{k:p[k] for k in ("sha256","rows","input_part_sha256") if k in p} for p in parts]}
        yield {"key":digest_json(identity), "start":parts[0]["start"],
            "rows":sum(p["rows"] for p in parts), "part_indices":list(range(start,start+len(parts))),
            "identity":identity}
    if cursor != store.manifest["input_count"] or cursor == 0:
        raise ValueError("Vector index requires complete, nonempty vector coverage.")


def _load(path):
    value = read_json(path)
    check_manifest(value)
    return value


def _faiss(config):
    import faiss
    faiss.omp_set_num_threads(min(config["threads"],os.cpu_count() or 1))
    return faiss


def build_faiss_index(store, output_dir, *, options=None, work_dir=None, max_new_shards=None):
    config = policy(options)
    if max_new_shards is not None and (type(max_new_shards) is not int or max_new_shards < 0):
        raise ValueError("Invalid vector index shard limit.")
    faiss = _faiss(config)
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=True)
    d, total = store.manifest["dimension"],store.manifest["input_count"]
    building = {k:config[k] for k in ("nlist","training_rows","training_iterations","threads")}
    # Small fixtures/pilots don't have enough samples to train IVF. This choice is
    # explicit in the manifest; large corpora never silently fall back to Flat.
    template_identity = {"encoder":store.manifest["encoder"], "dimension":d,
        "policy":building, "faiss":faiss.__version__, "format":"ivf-sq8-v1",
        "small_exact":total < 1024}
    template_key = digest_json(template_identity)
    template_dir = root / "templates" / template_key[:32]
    marker = template_dir / "template.json"
    if marker.exists():
        template = _load(marker)
        if template["identity"] != template_identity:
            raise ValueError("ANN template policy changed.")
        verify_file(template_dir / "trained.faiss",template["sha256"])
        trained = faiss.read_index(str(template_dir / "trained.faiss"))
    else:
        if total < 1024:
            trained = faiss.IndexFlatIP(d)
            sampled = 0
        else:
            sampled = min(total,config["training_rows"])
            nlist = min(config["nlist"],max(1,sampled//39))
            positions = np.sort(np.random.default_rng(1729).choice(total,sampled,replace=False))
            print(f"ANN training: {sampled:,} existing vectors; {nlist} IVF cells, SQ8",flush=True)
            values = np.ascontiguousarray(store.gather(positions),dtype=np.float32)
            validate_vectors(values,sampled,d)
            trained = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(d),d,nlist,
                faiss.ScalarQuantizer.QT_8bit,faiss.METRIC_INNER_PRODUCT)
            trained.cp.niter,trained.cp.seed = config["training_iterations"],1729
            trained.train(values)
            del values
        with local_workspace(work_dir) as temporary:
            local = Path(temporary) / "trained.faiss"
            faiss.write_index(trained,str(local))
            checksum = publish_file(local,template_dir / local.name)
        template = seal({"state":"COMPLETE", "identity":template_identity, "sha256":checksum,
            "trained_from":store.manifest["manifest_sha256"], "training_rows":sampled,
            "nlist":getattr(trained,"nlist",0), "ntotal":0})
        atomic_json(marker,template)
    if trained.d != d or trained.ntotal or not trained.is_trained:
        raise ValueError("Invalid ANN training template.")
    shards, written, reused = [],0,0
    for group in groups(store,config["parts_per_shard"]):
        key = digest_json({"group":group["key"], "template":template["manifest_sha256"]})
        shard_dir = root / "shards" / key[:32]
        marker = shard_dir / "shard.json"
        if marker.exists():
            saved = _load(marker)
            if saved["group"] != group["identity"] or saved["template"] != template["manifest_sha256"]:
                raise ValueError("ANN shard source binding changed.")
            verify_file(shard_dir / "vectors.faiss",saved["sha256"])
            reused += 1
        else:
            if max_new_shards is not None and written >= max_new_shards:
                break
            # Only one compressed shard and one source part are held in RAM.
            needed = group["rows"] * (d*(4 if total < 1024 else 1)+8) * 2 + 128*2**20
            if shutil.disk_usage(root).free < needed:
                raise RuntimeError("Insufficient local disk for the next ANN shard; restore checkpoints on a larger disk.")
            index = faiss.clone_index(trained)
            for i in group["part_indices"]:
                values = np.ascontiguousarray(store.part(i),dtype=np.float32)
                validate_vectors(values,store.manifest["parts"][i]["rows"],d)
                index.add(values)
            if index.ntotal != group["rows"]:
                raise ValueError("ANN shard dropped vector rows.")
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / "vectors.faiss"
                faiss.write_index(index,str(local))
                checksum = publish_file(local,shard_dir / local.name)
            del index
            saved = seal({"state":"COMPLETE", "group":group["identity"],
                "template":template["manifest_sha256"], "rows":group["rows"], "sha256":checksum})
            atomic_json(marker,saved)
            written += 1
        shards.append({"start":group["start"], "rows":group["rows"],
            "path":str((shard_dir / "vectors.faiss").relative_to(root)).replace("\\","/"),
            "sha256":saved["sha256"], "shard_sha256":saved["manifest_sha256"]})
        print(f"ANN index: {sum(s['rows'] for s in shards):,}/{total:,} vectors | new {written}, reused {reused}",flush=True)
    report = seal({"state":"COMPLETE" if sum(s["rows"] for s in shards) == total else "IN_PROGRESS",
        "backend":"faiss-flat-small-v1" if total < 1024 else "faiss-ivf-sq8-v1",
        "source":store.manifest["manifest_sha256"], "dimension":d, "rows":total,
        "template":template["manifest_sha256"], "nlist":template["nlist"], "shards":shards})
    atomic_json(root / "views" / (store.manifest["manifest_sha256"]+".json"),report)
    return report


class FaissVectorIndex:
    def __init__(self,root,manifest,config):
        check_manifest(manifest)
        self.root,self.manifest,self.config = Path(root),manifest,config
        self.verified = set()

    def search(self,vectors,k,effort):
        faiss = _faiss(self.config)
        scores = np.full((len(vectors),k),-np.inf,np.float32)
        positions = np.full((len(vectors),k),-1,np.int64)
        for shard in self.manifest["shards"]:
            path = bound(self.root,shard["path"])
            if path not in self.verified:
                verify_file(path,shard["sha256"])
                self.verified.add(path)
            # Windows FAISS wheels may omit OnDiskInvertedLists. Loading one
            # compressed shard is still bounded; Linux workers use mmap.
            flags = 0 if os.name == "nt" else faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            index = faiss.read_index(str(path),flags)
            if index.d != self.manifest["dimension"] or index.ntotal != shard["rows"]:
                raise ValueError("ANN shard shape mismatch.")
            probe = min(effort,self.manifest["nlist"]) if self.manifest["nlist"] else 0
            while True:
                if probe:
                    index.nprobe = probe
                ss,ii = index.search(np.ascontiguousarray(vectors),min(k,shard["rows"]))
                # Sparse cells can return fewer than k neighbors. Expand only
                # as necessary, and reject padding rather than dropping IDs.
                if not np.any(ii < 0) or probe >= self.manifest["nlist"]:
                    break
                probe = min(self.manifest["nlist"],max(1,probe*2))
            if np.any(ii < 0) or np.any(ii >= shard["rows"]):
                raise ValueError("Incomplete ANN shard results.")
            ii += shard["start"]
            scores,positions = merge(scores,positions,ss,ii,k)
            del index
        return scores,positions


def merge(scores,positions,new_scores,new_positions,k):
    scores,positions = np.concatenate((scores,new_scores),axis=1),np.concatenate((positions,new_positions),axis=1)
    order = np.lexsort((positions,-scores),axis=1)[:,:k]
    return np.take_along_axis(scores,order,axis=1),np.take_along_axis(positions,order,axis=1)


def refine_stream(store,queries,positions,output_dir,*,work_dir=None):
    """Rescore selected hits in ONE source-part pass, not one pass per query.

Especially important when the Qwen corpus does not fit the local cache: random
query-by-query gathers would repeatedly download entire remote vector blocks.
Checkpoint only selected scores per part (O(queries*k), not O(corpus*queries)).
"""
    if positions.ndim != 2 or positions.shape[0] != len(queries) or positions.dtype != np.int64:
        raise ValueError("Invalid ANN candidate positions.")
    if np.any(positions < 0) or np.any(positions >= store.manifest["input_count"]):
        raise ValueError("ANN candidate position outside source coverage.")
    if any(len(set(row.tolist())) != len(row) for row in positions):
        raise ValueError("ANN returned duplicate candidate positions.")
    root = Path(output_dir)
    signature = digest_json({"source":store.manifest["manifest_sha256"],
        "queries":hashlib.sha256(queries.tobytes()).hexdigest(),
        "positions":hashlib.sha256(positions.tobytes()).hexdigest(),"format":"stream-candidate-rescore-v1"})
    flat = positions.reshape(-1)
    order = np.argsort(flat,kind="stable")
    ordered = flat[order]
    scores = np.empty(flat.shape,np.float32)
    for i,part in enumerate(store.manifest["parts"]):
        low,high = np.searchsorted(ordered,[part["start"],part["start"]+part["rows"]])
        if low == high:
            continue
        selected = order[low:high]
        marker = root / f"part-{i:05d}.json"
        expected = {"signature":signature,"source_part":part["sha256"],"selected":int(high-low)}
        if marker.exists():
            saved = _load(marker)
            if any(saved.get(k) != v for k,v in expected.items()):
                raise ValueError("Vector refinement checkpoint changed.")
            verify_file(bound(root,saved["path"]),saved["sha256"])
            values = np.load(bound(root,saved["path"]),allow_pickle=False)
        else:
            block = store.part(i)
            values = np.empty(high-low,np.float32)
            for begin in range(0,len(selected),512):
                end = min(len(selected),begin+512)
                ids = selected[begin:end]
                source_rows = flat[ids]-part["start"]
                query_rows = ids//positions.shape[1]
                values[begin:end] = np.einsum("ij,ij->i",queries[query_rows],block[source_rows],optimize=True)
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / f"part-{i:05d}.npy"
                np.save(local,values,allow_pickle=False)
                checksum = publish_file(local,root / local.name)
            atomic_json(marker,seal({"state":"COMPLETE",**expected,"path":local.name,"sha256":checksum}))
        if values.shape != (high-low,) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid vector refinement scores.")
        scores[selected] = values
        if (i+1) % 32 == 0 or i+1 == len(store.manifest["parts"]):
            print(f"Vector refinement: {i+1}/{len(store.manifest['parts'])} source parts",flush=True)
    scores = scores.reshape(positions.shape)
    ranking = np.lexsort((positions,-scores),axis=1)
    return np.take_along_axis(scores,ranking,axis=1),np.take_along_axis(positions,ranking,axis=1)
