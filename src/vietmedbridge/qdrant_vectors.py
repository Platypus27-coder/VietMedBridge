"""Qdrant server backend; Python local mode is deliberately not used at scale.

Immutable content-addressed collections are shared by cumulative corpus views.
Integer point IDs are local shard positions; the view supplies the current
global offset. No raw medical text or credentials are stored in manifests.
"""
from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

import httpx
import numpy as np

from .artifacts import atomic_json, digest_json
from .content_embeddings import seal
from .embeddings import validate_vectors
from .vector_index import groups, merge, VectorIndexPending


class QdrantVectorIndex:
    def __init__(self,store,output_dir,config,*,client=None):
        self.config,self.store = config,store
        endpoint = os.environ.get("VMB_QDRANT_URL","").rstrip("/")
        url = urlsplit(endpoint)
        if url.scheme not in ("http","https") or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("Set VMB_QDRANT_URL to a Qdrant server URL without credentials/query parameters.")
        key = os.environ.get("VMB_QDRANT_API_KEY","")
        if key and url.scheme != "https" and url.hostname not in ("localhost","127.0.0.1","::1"):
            raise ValueError("Remote Qdrant API keys require HTTPS.")
        self.client = client or httpx.Client(base_url=endpoint,headers={"api-key":key} if key else {},timeout=180.)
        self.owns_client = client is None
        self.endpoint_digest = digest_json(endpoint)
        self.root = output_dir
        self.manifest = None

    def request(self,method,path,body=None,allow_missing=False):
        for attempt in range(5):
            try:
                response = self.client.request(method,path,json=body)
            except (httpx.TimeoutException,httpx.NetworkError):
                if attempt == 4:
                    raise RuntimeError("Qdrant network request failed; acknowledged parts remain reusable.") from None
            else:
                if allow_missing and response.status_code == 404:
                    return None
                if response.status_code not in (429,502,503,504):
                    if response.is_error:
                        # Headers/body can contain private endpoint details.
                        raise RuntimeError(f"Qdrant {method} request failed with HTTP {response.status_code}.")
                    data = response.json()
                    if data.get("status") != "ok":
                        raise RuntimeError("Qdrant did not confirm the operation.")
                    return data["result"]
                if attempt == 4:
                    raise RuntimeError("Qdrant is rate limited/unavailable; retry with the same configuration.")
            time.sleep(min(2**attempt,8))

    def prepare(self):
        store,config = self.store,self.config
        shards, pending, observations = [],[],[]
        for group in groups(store,config["parts_per_shard"]):
            # Explicit version fixes immutable collection settings and ownership.
            name = "vmb_"+digest_json({"source":group["key"],"format":"qdrant-cosine-ondisk-hnsw-v1"})[:48]
            base = "/collections/"+name
            info = self.request("GET",base,allow_missing=True)
            if info is None:
                self.request("PUT",base,{"vectors":{"size":store.manifest["dimension"],"distance":"Cosine","on_disk":True},
                    "on_disk_payload":True,"hnsw_config":{"m":16,"ef_construct":100,"on_disk":True},
                    "optimizers_config":{"indexing_threshold":1000,"max_optimization_threads":2}})
                self.request("PUT",base+"/index?wait=true",{"field_name":"vmb_part","field_schema":"keyword"})
                info = self.request("GET",base)
            params = info["config"]["params"]["vectors"]
            if params.get("size") != store.manifest["dimension"] or params.get("distance") != "Cosine" or params.get("on_disk") is not True:
                raise ValueError("Qdrant collection vector policy mismatch; no collection was deleted.")
            for i in group["part_indices"]:
                part = store.manifest["parts"][i]
                part_key = digest_json({"group":group["key"],"offset":part["start"]-group["start"],
                    "sha256":part["sha256"],"rows":part["rows"]})
                predicate = {"must":[{"key":"vmb_part","match":{"value":part_key}}]}
                count = self.request("POST",base+"/points/count",{"exact":True,"filter":predicate})["count"]
                if count != part["rows"]:
                    values = np.ascontiguousarray(store.part(i),dtype=np.float32)
                    validate_vectors(values,part["rows"],store.manifest["dimension"])
                    for start in range(0,part["rows"],config["qdrant_batch_size"]):
                        end = min(part["rows"],start+config["qdrant_batch_size"])
                        ids = list(range(part["start"]-group["start"]+start,part["start"]-group["start"]+end))
                        result = self.request("PUT",base+"/points?wait=true",{"batch":{"ids":ids,
                            "vectors":values[start:end].tolist(),"payloads":[{"vmb_part":part_key,"vmb_group":group["key"]} for _ in ids]}})
                        if result.get("status") != "completed":
                            raise RuntimeError("Qdrant has not durably acknowledged a vector batch.")
                    if self.request("POST",base+"/points/count",{"exact":True,"filter":predicate})["count"] != part["rows"]:
                        raise ValueError("Qdrant part row count mismatch after upsert.")
                print(f"Qdrant: {part['start']+part['rows']:,}/{store.manifest['input_count']:,} vectors acknowledged",flush=True)
            if self.request("POST",base+"/points/count",{"exact":True})["count"] != group["rows"]:
                raise ValueError("Qdrant collection contains missing or unexpected points.")
            # Ingestion completion is distinct from HNSW optimization. Don't
            # label a million-row unindexed collection ANN-ready.
            info = self.request("GET",base)
            status = str(info.get("status","")).lower()
            if info.get("optimizer_status") != "ok" or status != "green" or (group["rows"] >= 10000 and info.get("indexed_vectors_count",0) == 0):
                pending.append({"collection":name,"rows":group["rows"],
                    "indexed_vectors_count":info.get("indexed_vectors_count",0)})
            shards.append({"collection":name,"start":group["start"],"rows":group["rows"],"group":group["key"],
                "distance":"Cosine"})
            observations.append({"collection":name,"status":status,"indexed_vectors_count":info.get("indexed_vectors_count",0)})
        if pending:
            progress = seal({"state":"WAITING_FOR_QDRANT_OPTIMIZATION","pending_collections":pending,
                "acknowledged_rows":sum(s["rows"] for s in shards),
                "scope":"Vectors acknowledged; rerun after background HNSW optimization. No reencoding."})
            atomic_json(self.root / "qdrant_progress.json",progress)
            raise VectorIndexPending(progress)
        self.manifest = seal({"state":"COMPLETE","backend":"qdrant-server-hnsw-v1",
            "endpoint_sha256":self.endpoint_digest,"source":store.manifest["manifest_sha256"],
            "dimension":store.manifest["dimension"],"rows":store.manifest["input_count"],"shards":shards})
        atomic_json(self.root / "views" / (store.manifest["manifest_sha256"]+".json"),self.manifest)
        atomic_json(self.root / "qdrant_status.json",{"collections":observations,"state":"COMPLETE"})
        return self.manifest

    def search(self,vectors,k,effort):
        scores = np.full((len(vectors),k),-np.inf,np.float32)
        positions = np.full((len(vectors),k),-1,np.int64)
        for shard in self.manifest["shards"]:
            response = self.request("POST","/collections/"+shard["collection"]+"/points/query/batch",
                {"searches":[{"query":q.tolist(),"limit":min(k,shard["rows"]),"with_payload":["vmb_group"],
                    "params":{"hnsw_ef":max(effort,self.config["qdrant_hnsw_ef"]),"exact":False}} for q in vectors]})
            if len(response) != len(vectors):
                raise ValueError("Qdrant query batch response count mismatch.")
            ss,ii = [],[]
            for result in response:
                points = result["points"]
                if len(points) != min(k,shard["rows"]) or any(type(p["id"]) is not int or not 0 <= p["id"] < shard["rows"]
                    or p.get("payload",{}).get("vmb_group") != shard["group"] for p in points):
                    raise ValueError("Qdrant returned missing/foreign vector IDs.")
                ss.append([p["score"] for p in points])
                ii.append([p["id"]+shard["start"] for p in points])
            scores,positions = merge(scores,positions,np.asarray(ss,np.float32),np.asarray(ii,np.int64),k)
        return scores,positions

    def close(self):
        if self.owns_client:
            self.client.close()
