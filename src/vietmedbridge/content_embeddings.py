"""Content keyed vector blocks shared by corpus versions; bounded local lookups."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, local_workspace, publish_file, read_json, verify_file
from .embeddings import validate_vectors
from .scale_vectors import bound, check_manifest


def seal(value):
    value = dict(value)
    value["manifest_sha256"] = digest_json(value)
    return value


def checked(value):
    if digest_json({k:v for k,v in value.items() if k != "manifest_sha256"}) != value.get("manifest_sha256"):
        raise ValueError("Content cache metadata changed.")
    return value


def key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ContentBlocks:
    def __init__(self, data_root, encoder, work_dir):
        self.root, self.encoder = Path(data_root).resolve(), encoder
        self.identity = {"encoder":encoder.identity,"dimension":encoder.dimension,"format":"content-float32-l2-v1"}
        self.directory = self.root / "model_cache" / digest_json(self.identity)[:24]
        self.directory.mkdir(parents=True,exist_ok=True)
        Path(work_dir).mkdir(parents=True,exist_ok=True)
        # Each runtime owns its lookup. Only immutable block files are shared.
        self.temporary = local_workspace(work_dir)
        self.local = Path(self.temporary.__enter__())
        self.con = sqlite3.connect(self.local / "lookup.sqlite")
        self.con.execute("CREATE TABLE vectors(key TEXT PRIMARY KEY,path TEXT,sha TEXT,row INTEGER)")
        for path in sorted(self.directory.glob("*.done.json")):
            self.register(checked(read_json(path)))

    def close(self):
        self.con.close()
        self.temporary.__exit__(None,None,None)

    def register(self, block):
        if block["identity"] != self.identity or len(block["keys"]) != block["rows"]:
            raise ValueError("Content vector producer/key mismatch.")
        bound(self.root,block["path"])
        self.con.executemany("INSERT OR IGNORE INTO vectors VALUES (?,?,?,?)",
            [(k,block["path"],block["sha256"],i) for i,k in enumerate(block["keys"])])

    def save_block(self, keys, path, checksum):
        value = seal({"identity":self.identity,"keys":keys,"rows":len(keys),
            "path":Path(path).resolve().relative_to(self.root).as_posix(),"sha256":checksum})
        marker = self.directory / ("block-"+digest_json(keys)+".done.json")
        if marker.exists():
            old = checked(read_json(marker))
            if old["identity"] != self.identity or old["keys"] != keys:
                raise ValueError("Content block identity changed.")
            # Existing immutable vectors remain authoritative across batch sizes.
            verify_file(bound(self.root,old["path"]),old["sha256"])
            value = old
        else:
            atomic_json(marker,value)
        self.register(value)

    def seed(self, input_root, inputs, vector_root):
        """Register the already paid-for 04 baseline blocks without copying vectors."""
        check_manifest(inputs)
        vector_root = Path(vector_root)
        if not (vector_root / "config.json").exists():
            return 0
        config = read_json(vector_root / "config.json")
        if digest_json({k:v for k,v in config.items() if k != "signature"}) != config.get("signature"):
            raise ValueError("Baseline seed configuration changed.")
        if config.get("encoder") != self.encoder.identity or config.get("dimension") != self.encoder.dimension:
            return 0
        if config.get("input_manifest_sha256") != inputs["manifest_sha256"]:
            return 0
        count = 0
        for source in inputs["parts"]:
            stem = f"part-{source['start']:010d}-{source['rows']:05d}"
            marker = vector_root / (stem+".done.json")
            if not marker.exists():
                continue
            saved = read_json(marker)
            if any(saved.get(k) != v for k,v in {"signature":config["signature"],"start":source["start"],
                "rows":source["rows"],"input_part_sha256":source["sha256"],"path":stem+".npy"}.items()):
                raise ValueError("Baseline seed does not match frozen input order.")
            path = bound(vector_root,saved["path"])
            verify_file(path,saved["sha256"])
            validate_vectors(np.load(path,allow_pickle=False,mmap_mode="r"),source["rows"],self.encoder.dimension)
            source_path = bound(input_root,source["path"])
            verify_file(source_path,source["sha256"])
            texts = pq.read_table(source_path,columns=["text"]).column("text").to_pylist()
            keys = [key(t) for t in texts]
            self.save_block(keys,path,saved["sha256"])
            count += len(keys)
        return count

    def refs(self, texts, *, batch_size, encode_missing):
        keys = [key(t) for t in texts]
        missing = {}
        for k,text in zip(keys,texts,strict=True):
            if self.con.execute("SELECT 1 FROM vectors WHERE key=?",(k,)).fetchone() is None:
                missing[k] = text
        if missing and not encode_missing:
            return None,0
        if missing:
            ordered = sorted(missing,key=lambda k:(len(missing[k]),k))
            values = np.asarray(self.encoder.encode([missing[k] for k in ordered],batch_size=batch_size),np.float32)
            validate_vectors(values,len(ordered),self.encoder.dimension)
            local = self.local / "new-vectors.npy"
            np.save(local,values,allow_pickle=False)
            from .artifacts import sha256_file
            # Two producers may encounter overlapping text. Never overwrite a
            # previously published vector file, even if batching changes bits.
            path = self.directory / ("block-"+sha256_file(local)+".npy")
            checksum = publish_file(local,path)
            self.save_block(ordered,path,checksum)
        refs = [self.con.execute("SELECT path,sha,row FROM vectors WHERE key=?",(k,)).fetchone() for k in keys]
        return [{"path":p,"sha256":s,"row":r} for p,s,r in refs],len(missing)


def content_embeddings(data_root,input_root,inputs,encoder,output_dir,*,work_dir,batch_size=32,
                       max_new_parts=None,worker_id=None,workers=1,encode_missing=True,seed_root=None,seed_sources=(),
                       should_stop=None,on_part=None):
    check_manifest(inputs)
    if (type(workers) is not int or workers < 1 or (worker_id is not None and
        (type(worker_id) is not int or not 0 <= worker_id < workers)) or type(batch_size) is not int or batch_size < 1
        or (max_new_parts is not None and (type(max_new_parts) is not int or max_new_parts < 0))):
        raise ValueError("Invalid content embedding worker/limits.")
    root = Path(output_dir)
    identity = {"input_manifest_sha256":inputs["manifest_sha256"],"input_count":inputs["input_count"],
        "encoder":encoder.identity,"dimension":encoder.dimension,"format":"content-reference-v1"}
    signature = digest_json(identity)
    if (root / "config.json").exists() and read_json(root / "config.json") != identity:
        raise ValueError("Content embedding inputs/model changed; use a new view.")
    # Coordinator initializes the view; workers never rewrite a shared manifest.
    if worker_id is None:
        atomic_json(root / "config.json",identity)
    blocks = ContentBlocks(data_root,encoder,work_dir)
    parts,new_parts,new_texts,reused_texts = [],0,0,0
    try:
        if seed_root is not None:
            blocks.seed(input_root,inputs,seed_root)
        for source_root,source_inputs,vector_root in seed_sources:
            blocks.seed(source_root,source_inputs,vector_root)
        cursor, stopped = 0, False
        for index,source in enumerate(inputs["parts"]):
            if source["start"] != cursor or source["rows"] < 1:
                raise ValueError("Content input ordering changed.")
            cursor += source["rows"]
            marker = root / f"map-{source['start']:012d}.done.json"
            expected = {"signature":signature,"start":source["start"],"rows":source["rows"],"input_sha256":source["sha256"]}
            assigned = worker_id is None or index % workers == worker_id
            if marker.exists():
                saved = checked(read_json(marker))
                if any(saved.get(k) != v for k,v in expected.items()):
                    raise ValueError("Content mapping checkpoint changed.")
                verify_file(bound(root,saved["path"]),saved["sha256"])
                reused_texts += source["rows"]
            elif assigned and not stopped and (max_new_parts is None or new_parts < max_new_parts):
                if should_stop is not None and should_stop(source):
                    stopped = True
                    continue
                path = bound(input_root,source["path"])
                verify_file(path,source["sha256"])
                texts = pq.read_table(path,columns=["text"]).column("text").to_pylist()
                if len(texts) != source["rows"] or any(not isinstance(t,str) or not t.strip() for t in texts):
                    raise ValueError("Invalid content embedding input.")
                refs,count = blocks.refs(texts,batch_size=batch_size,encode_missing=encode_missing)
                if refs is None:
                    continue
                local = blocks.local / "mapping.parquet"
                pq.write_table(pa.Table.from_pylist(refs),local,compression="zstd")
                target = root / f"map-{source['start']:012d}.parquet"
                saved = seal(expected | {"path":target.name,"sha256":publish_file(local,target)})
                atomic_json(marker,saved)
                new_parts += 1
                new_texts += count
                reused_texts += source["rows"]-count
            else:
                continue
            parts.append(saved)
            if assigned:
                if on_part is not None:
                    on_part(saved)
                print(f"{encoder.identity.get('model_id')} | new texts {new_texts:,}, reused {reused_texts:,}, checked parts {len(parts)}/{len(inputs['parts'])}",flush=True)
        if cursor != inputs["input_count"]:
            raise ValueError("Content input count changed.")
        result = seal(identity | {"signature":signature,"parts":parts,
            "state":"COMPLETE" if len(parts) == len(inputs["parts"]) else "IN_PROGRESS"})
        report_name = "embeddings.json" if worker_id is None else f"worker-{worker_id}.json"
        atomic_json(root / report_name,result)
        atomic_json(root / ("profile.json" if worker_id is None else f"worker-{worker_id}-profile.json"),
            {"new_texts":new_texts,"reused_texts":reused_texts,"new_parts":new_parts,"worker_id":worker_id,"workers":workers})
        return result
    finally:
        blocks.close()


class ContentVectorParts:
    """The same bounded search/gather interface as VectorParts, with shared blocks."""
    def __init__(self,data_root,root,manifest,cache_dir,*,local_cache_bytes=4*2**30):
        check_manifest(manifest)
        self.data_root,self.root,self.manifest = Path(data_root),Path(root),manifest
        if type(local_cache_bytes) is not int or local_cache_bytes < 1:
            raise ValueError("Invalid local content cache budget.")
        self.cache = Path(cache_dir) / digest_json([str(self.root),manifest["manifest_sha256"]])[:16]
        self.cache.mkdir(parents=True,exist_ok=True)
        self.cache_budget = local_cache_bytes
        self.disk = OrderedDict((p.name,(p,p.stat().st_size)) for p in sorted(self.cache.iterdir(),key=lambda p:p.stat().st_mtime)
            if p.suffix in (".npy",".parquet"))
        self.disk_bytes = sum(s for _,s in self.disk.values())
        self.maps,self.refs,self.verified = OrderedDict(),OrderedDict(),set()
        self.starts = [p["start"] for p in manifest["parts"]]
        cursor = 0
        for p in manifest["parts"]:
            if p["start"] != cursor or p["rows"] < 1 or p["signature"] != manifest["signature"]:
                raise ValueError("Content vector coverage changed.")
            cursor += p["rows"]
        if cursor != manifest["input_count"]:
            raise ValueError("Incomplete content vector coverage.")

    def _touch(self,path):
        previous = self.disk.pop(path.name,None)
        if previous:
            self.disk_bytes -= previous[1]
        size = path.stat().st_size
        self.disk[path.name] = path,size
        self.disk_bytes += size
        for name,(old,bytes_) in list(self.disk.items()):
            if self.disk_bytes <= self.cache_budget:
                break
            if old.stem in self.maps:  # at most four live vector blocks
                continue
            old.unlink(missing_ok=True)  # generated local copy, never Drive data
            self.disk_bytes -= bytes_
            del self.disk[name]
            self.verified.discard(old.stem)

    def _block(self,ref):
        name = digest_json([ref["path"],ref["sha256"]])
        if name not in self.maps:
            path = self.cache / (name+".npy")
            if not path.exists():
                source = bound(self.data_root,ref["path"])
                verify_file(source,ref["sha256"])
                publish_file(source,path)
            if name not in self.verified:
                verify_file(path,ref["sha256"])
                self.verified.add(name)
            values = np.load(path,allow_pickle=False,mmap_mode="r")
            validate_vectors(values,len(values),self.manifest["dimension"])
            self.maps[name] = values
            if len(self.maps) > 4:
                self.maps.popitem(last=False)
        self.maps.move_to_end(name)
        self._touch(self.cache / (name+".npy"))
        return self.maps[name]

    def _references(self,index):
        if index not in self.refs:
            p = self.manifest["parts"][index]
            name = digest_json([str(self.root),p["path"],p["sha256"]])
            path = self.cache / (name+".parquet")
            if not path.exists():
                source = bound(self.root,p["path"])
                verify_file(source,p["sha256"])
                publish_file(source,path)
            if name not in self.verified:
                verify_file(path,p["sha256"])
                self.verified.add(name)
            refs = pq.read_table(path).to_pylist()
            if len(refs) != p["rows"]:
                raise ValueError("Content map shape changed.")
            self.refs[index] = refs
            if len(self.refs) > 4:
                self.refs.popitem(last=False)
            self._touch(path)
        self.refs.move_to_end(index)
        return self.refs[index]

    def gather(self,positions):
        positions = np.asarray(positions,np.int64)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= self.manifest["input_count"]):
            raise ValueError("Invalid content vector positions.")
        groups = {}
        for out in np.argsort(positions,kind="stable"):
            position = positions[out]
            index = bisect_right(self.starts,int(position))-1
            ref = self._references(index)[int(position)-self.starts[index]]
            groups.setdefault((ref["path"],ref["sha256"]),[]).append((out,ref["row"]))
        result = np.empty((len(positions),self.manifest["dimension"]),np.float32)
        for (path,sha),rows in groups.items():
            block = self._block({"path":path,"sha256":sha})
            outputs,offsets = zip(*rows)
            if min(offsets) < 0 or max(offsets) >= len(block):
                raise ValueError("Content map row escapes its vector block.")
            result[list(outputs)] = block[list(offsets)]
        return result

    def part(self,index):
        p = self.manifest["parts"][index]
        return self.gather(range(p["start"],p["start"]+p["rows"]))
