"""Stream frozen input parts to durable vectors; search all parts in bounded blocks."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, verify_file
from .embeddings import unit_signature, validate_vectors


def bound(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Vector/input path escapes its directory.")
    return path


def check_manifest(value):
    if (value.get("state") != "COMPLETE" or
        digest_json({k:v for k,v in value.items() if k != "manifest_sha256"}) != value.get("manifest_sha256")):
        raise ValueError("A complete, unchanged manifest is required.")


def stream_embeddings(input_root, inputs, encoder, output_dir, *, batch_size=32,
                      work_dir=None, max_new_parts=None):
    check_manifest(inputs)
    if type(batch_size) is not int or batch_size < 1 or (max_new_parts is not None and
        (type(max_new_parts) is not int or max_new_parts < 0)):
        raise ValueError("Invalid streamed embedding limits.")
    root = Path(output_dir)
    identity = {"input_manifest_sha256":inputs["manifest_sha256"], "input_count":inputs["input_count"],
        "encoder":encoder.identity, "dimension":encoder.dimension, "format":"float32-unit-vectors-v1",
        "order":"frozen-input-part-order", "units_sha256":inputs["manifest_sha256"]}
    signature = digest_json(identity)
    config = identity | {"signature":signature}
    if (root / "config.json").exists() and read_json(root / "config.json") != config:
        raise ValueError("Streamed input/model policy changed; choose a new vector directory.")
    atomic_json(root / "config.json", config)
    parts, cursor, written, reused = [], 0, 0, 0
    started = time.perf_counter()
    for source in inputs["parts"]:
        if source["start"] != cursor or source["rows"] <= 0:
            raise ValueError("Input parts are out of order.")
        stem = f"part-{cursor:010d}-{source['rows']:05d}"
        expected = {"signature":signature,"start":cursor,"rows":source["rows"],
            "path":stem+".npy", "input_part_sha256":source["sha256"]}
        marker = root / (stem+".done.json")
        if marker.exists():
            saved = read_json(marker)
            if any(saved.get(k) != v for k,v in expected.items()):
                raise ValueError("Streamed vector checkpoint/input mismatch.")
            path = bound(root,saved["path"])
            verify_file(path,saved["sha256"])
            validate_vectors(np.load(path,allow_pickle=False,mmap_mode="r"),source["rows"],encoder.dimension)
            reused += 1
        else:
            if max_new_parts is not None and written >= max_new_parts:
                break
            path = bound(input_root,source["path"])
            verify_file(path,source["sha256"])
            rows = pq.read_table(path,columns=["id","text"]).to_pylist()
            if len(rows) != source["rows"]:
                raise ValueError("Input part row count changed.")
            unit_signature(rows)  # Reject duplicate IDs and blank inputs before GPU work.
            # Similar lengths reduce padding. Restore the immutable input order afterwards.
            order = sorted(range(len(rows)),key=lambda i:(len(rows[i]["text"]),rows[i]["id"]))
            tick = time.perf_counter()
            values = np.asarray(encoder.encode([rows[i]["text"] for i in order],batch_size=batch_size),np.float32)
            validate_vectors(values,len(rows),encoder.dimension)
            vectors = np.empty_like(values)
            vectors[order] = values
            with local_workspace(work_dir) as temporary:
                local = Path(temporary) / expected["path"]
                np.save(local,vectors,allow_pickle=False)
                checksum = publish_file(local,root / local.name)
            saved = expected | {"sha256":checksum,"encoding_seconds":round(time.perf_counter()-tick,3)}
            atomic_json(marker,saved)
            written += 1
        parts.append(saved)
        cursor += source["rows"]
        print(f"BGE vectors: {cursor:,}/{inputs['input_count']:,} | new {written}, reused {reused}",flush=True)
    report = identity | {"signature":signature,"parts":parts,
        "state":"COMPLETE" if cursor == inputs["input_count"] else "IN_PROGRESS"}
    report["manifest_sha256"] = digest_json(report)
    atomic_json(root / "embeddings.json",report)
    atomic_json(root / "runtime_profile.json", {"new_parts":written,"reused_parts":reused,
        "batch_size_requested":batch_size,"safe_batch_size":getattr(encoder,"safe_batch_size",batch_size),
        "oom_backoffs":getattr(encoder,"oom_backoffs",0),"seconds_this_call":time.perf_counter()-started})
    return report


class VectorParts:
    """Local, verified memory maps; never concatenate the corpus vector matrix."""
    def __init__(self, root, manifest, cache_dir):
        check_manifest(manifest)
        self.root, self.manifest, self.cache = Path(root),manifest,Path(cache_dir)
        self.cache = self.cache / manifest["manifest_sha256"][:16]
        self.cache.mkdir(parents=True,exist_ok=True)
        self.starts, self.maps, self.verified, cursor = [],OrderedDict(),set(),0
        for p in manifest["parts"]:
            if p["start"] != cursor or p["rows"] <= 0 or p["signature"] != manifest["signature"]:
                raise ValueError("Vector part order/signature mismatch.")
            self.starts.append(cursor)
            cursor += p["rows"]
        if cursor != manifest["input_count"]:
            raise ValueError("Incomplete vector coverage.")

    def part(self,index):
        p = self.manifest["parts"][index]
        if index not in self.maps:
            local = bound(self.cache,p["path"])
            if not local.is_file():
                verify_file(bound(self.root,p["path"]),p["sha256"])
                publish_file(bound(self.root,p["path"]),local)
            if index not in self.verified:
                verify_file(local,p["sha256"])
                self.verified.add(index)
            values = np.load(local,allow_pickle=False,mmap_mode="r")
            validate_vectors(values,p["rows"],self.manifest["dimension"])
            self.maps[index] = values
            if len(self.maps) > 4:
                self.maps.popitem(last=False)
        self.maps.move_to_end(index)
        return self.maps[index]

    def gather(self,positions):
        positions = np.asarray(positions,dtype=np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= self.manifest["input_count"]):
            raise ValueError("Invalid vector positions.")
        result = np.empty((len(positions),self.manifest["dimension"]),np.float32)
        groups = {}
        for out,pos in enumerate(positions):
            index = bisect_right(self.starts,int(pos))-1
            groups.setdefault(index,[]).append((out,int(pos)-self.starts[index]))
        for index,rows in groups.items():
            outputs,offsets = zip(*rows)
            result[list(outputs)] = self.part(index)[list(offsets)]
        return result


def search_parts(store, query_vectors, output_dir, *, k=2048, query_batch_size=128,
                 work_dir=None, device="cuda"):
    """Exact child cosine top-k. GPU/CPU blocks scan every frozen input part once."""
    validate_vectors(query_vectors,len(query_vectors),store.manifest["dimension"])
    if type(k) is not int or k < 1 or type(query_batch_size) is not int or query_batch_size < 1:
        raise ValueError("Invalid search limits.")
    k = min(k,store.manifest["input_count"])
    root = Path(output_dir)
    identity = {"embeddings":store.manifest["manifest_sha256"],
        "queries_sha256":hashlib.sha256(query_vectors.tobytes()).hexdigest(),
        "k":k,"backend":"exact-part-cosine-v1","device":device,"numpy":np.__version__}
    producer = {}
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for corpus-wide dense search.")
        identity.update(matmul="float32-tf32-disabled")
        producer = {"torch":torch.__version__,"cuda_device":torch.cuda.get_device_name(0)}
    elif device != "cpu":
        raise ValueError("Unsupported dense search device.")
    signature = digest_json(identity)
    marker = root / "search.json"
    if marker.exists():
        saved = read_json(marker)
        check_manifest(saved)
        if saved["signature"] != signature:
            raise ValueError("Dense search checkpoint policy changed.")
        for key in ("scores","positions"):
            verify_file(bound(root,saved[key]["path"]),saved[key]["sha256"])
        return np.load(root / saved["scores"]["path"],allow_pickle=False),np.load(root / saved["positions"]["path"],allow_pickle=False),saved
    scores = np.full((len(query_vectors),k),-np.inf,np.float32)
    positions = np.full((len(query_vectors),k),-1,np.int64)
    for index,p in enumerate(store.manifest["parts"]):
        values = np.asarray(store.part(index))
        block = torch.from_numpy(values.copy()).to("cuda") if device == "cuda" else values
        for start in range(0,len(query_vectors),query_batch_size):
            q = query_vectors[start:start+query_batch_size]
            if device == "cuda":
                previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    with torch.inference_mode():
                        sims = torch.from_numpy(q).to("cuda") @ block.T
                        # Stable ties match the immutable vector position order.
                        ii = torch.argsort(sims,dim=1,descending=True,stable=True)[:,:min(k,p["rows"])]
                        ss = torch.gather(sims,1,ii)
                        ss,ii = ss.cpu().numpy(),ii.cpu().numpy().astype(np.int64)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            else:
                sims = q @ block.T
                ii = np.argsort(-sims,axis=1,kind="stable")[:,:k]
                ss = np.take_along_axis(sims,ii,axis=1)
            ii += p["start"]
            old = slice(start,start+len(q))
            ss = np.concatenate([scores[old],ss],axis=1)
            ii = np.concatenate([positions[old],ii],axis=1)
            # Stable position tie-break for scores retained by the block top-k.
            order = np.lexsort((ii,-ss),axis=1)[:,:k]
            scores[old],positions[old] = np.take_along_axis(ss,order,axis=1),np.take_along_axis(ii,order,axis=1)
        if device == "cuda":
            del block
        print(f"Dense search: {index+1}/{len(store.manifest['parts'])} parts, all {len(query_vectors)} queries",flush=True)
    if np.any(positions < 0) or not np.isfinite(scores).all():
        raise ValueError("Incomplete corpus-wide dense search.")
    saved = identity | {"signature":signature,"state":"COMPLETE","producer":producer}
    with local_workspace(work_dir) as temporary:
        for key,values in (("scores",scores),("positions",positions)):
            local = Path(temporary) / (key+".npy")
            np.save(local,values,allow_pickle=False)
            saved[key] = {"path":local.name,"sha256":publish_file(local,root / local.name)}
    saved["manifest_sha256"] = digest_json(saved)
    atomic_json(marker,saved)
    return scores,positions,saved
