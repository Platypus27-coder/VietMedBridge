"""Real FAISS IVF/SQ8 persistence, append reuse, recall and restart checks."""
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from vietmedbridge.artifacts import digest_json, read_json
from vietmedbridge.content_embeddings import seal
from vietmedbridge.scale_vectors import search_parts
from vietmedbridge.vector_index import build_faiss_index, policy
from vietmedbridge.vector_search import search_vectors
from vietmedbridge.qdrant_vectors import QdrantVectorIndex
from vietmedbridge.vector_index import FaissVectorIndex, refine_stream


class Store:
    def __init__(self,values,part_size=256):
        self.values = np.asarray(values,np.float32)
        self.calls,self.gathers = [],[]
        parts = [{"start":start,"rows":len(values[start:start+part_size]),
            "sha256":digest_json(values[start:start+part_size].tolist())} for start in range(0,len(values),part_size)]
        self.manifest = seal({"state":"COMPLETE","dimension":values.shape[1],"encoder":{"model":"synthetic-normalized-v1"},
            "input_count":len(values),"parts":parts})
    def part(self,index):
        self.calls.append(index)
        p = self.manifest["parts"][index]
        return self.values[p["start"]:p["start"]+p["rows"]]
    def gather(self,positions):
        positions = list(positions)
        self.gathers.append(positions)
        return self.values[positions]


def normalized(n=1536,d=32,seed=73):
    values = np.random.default_rng(seed).normal(size=(n,d)).astype(np.float32)
    return values/np.linalg.norm(values,axis=1,keepdims=True)


OPTIONS = {"parts_per_shard":2,"nlist":16,"training_rows":1024,"training_iterations":3,
    "threads":1,"nprobe":8,"max_nprobe":16,"query_batch_size":2,"recall_queries":4}


def test_real_ivf_resume_append_no_reencoding_and_corruption(tmp_path):
    data = normalized(2048)
    first = Store(data[:1536])
    partial = build_faiss_index(first,tmp_path,options=OPTIONS,max_new_shards=1)
    assert partial["state"] == "IN_PROGRESS" and first.calls == [0,1]
    first.calls.clear()
    full = build_faiss_index(first,tmp_path,options=OPTIONS)
    assert full["state"] == "COMPLETE" and first.calls == [2,3,4,5]
    assert full["backend"] == "faiss-ivf-sq8-v1"
    newer = Store(data)
    appended = build_faiss_index(newer,tmp_path,options=OPTIONS)
    assert newer.calls == [6,7] and not newer.gathers  # no retraining either
    assert full["shards"] == appended["shards"][:3]
    assert sum(s["rows"] for s in appended["shards"]) == 2048
    broken = tmp_path / appended["shards"][0]["path"]
    broken.write_bytes(b"corrupt index")
    with pytest.raises(ValueError,match="changed artifact"):
        build_faiss_index(newer,tmp_path,options=OPTIONS)


def test_ann_exact_recall_query_checkpoint_and_complete_reuse(tmp_path,monkeypatch):
    store = Store(normalized())
    queries = normalized(7,32,51)
    kwargs = {"k":128,"device":"cpu","options":OPTIONS,"index_root":tmp_path / "index"}
    partial = search_vectors(store,queries,tmp_path / "search",max_new_batches=1,**kwargs)
    assert partial[2]["completed_queries"] == 2
    from vietmedbridge import vector_index
    original = vector_index.FaissVectorIndex.search
    visits = []
    monkeypatch.setattr(vector_index.FaissVectorIndex,"search",lambda self,q,*a:(visits.append(len(q)),original(self,q,*a))[1])
    ss,ii,report = search_vectors(store,queries,tmp_path / "search",**kwargs)
    assert visits == [2,2,1]  # first query batch AND quality gate were checkpointed
    truth = search_parts(store,queries,tmp_path / "truth",k=20,device="cpu")[1]
    recall = np.mean([len(set(a)&set(b[:20]))/20 for a,b in zip(truth,ii,strict=True)])
    assert recall >= .95 and report["quality"]["recall"] >= .95
    np.testing.assert_allclose(ss,np.take_along_axis(queries @ store.values.T,ii,axis=1),atol=1e-6)
    store.calls.clear();store.gathers.clear();visits.clear()
    repeated = search_vectors(store,queries,tmp_path / "search",**kwargs)
    assert repeated[2] == report and not visits and not store.calls and not store.gathers


def test_gate_failure_prevents_final_query_results(tmp_path,monkeypatch):
    from vietmedbridge.vector_index import FaissVectorIndex
    store = Store(normalized())
    queries = store.values[-3:]
    def wrong(self,values,k,effort):
        return np.zeros((len(values),k),np.float32),np.tile(np.arange(k,dtype=np.int64),(len(values),1))
    monkeypatch.setattr(FaissVectorIndex,"search",wrong)
    with pytest.raises(RuntimeError,match="recall gate failed"):
        search_vectors(store,queries,tmp_path / "search",k=20,device="cpu",options=OPTIONS)
    assert list((tmp_path / "search").rglob("ann_quality_failed.json"))
    assert not list((tmp_path / "search").glob("*/search.json"))


def test_reused_shards_with_new_global_offsets_and_single_pass_refinement(tmp_path):
    data = normalized(1536)
    store = Store(data)
    before = build_faiss_index(store,tmp_path / "index",options=OPTIONS)
    newer = Store(np.concatenate((normalized(512,32,19),data)))
    after = build_faiss_index(newer,tmp_path / "index",options=OPTIONS)
    assert newer.calls == [0,1] and not newer.gathers
    assert [s["path"] for s in before["shards"]] == [s["path"] for s in after["shards"][1:]]
    assert [s["start"]+512 for s in before["shards"]] == [s["start"] for s in after["shards"][1:]]
    queries = data[-3:]
    index = FaissVectorIndex(tmp_path / "index",after,policy(OPTIONS))
    _,positions = index.search(queries,128,16)
    newer.calls.clear()
    ss,ii = refine_stream(newer,queries,positions,tmp_path / "refinement")
    assert ii[:,0].tolist() == [2045,2046,2047]
    assert len(newer.calls) == len(set(newer.calls))  # each selected source part only once
    newer.calls.clear()
    again = refine_stream(newer,queries,positions,tmp_path / "refinement")
    assert not newer.calls
    np.testing.assert_array_equal(ii,again[1])
    np.testing.assert_allclose(ss,again[0])


class Server:
    """REST contract test server; does not claim to benchmark Qdrant HNSW."""
    def __init__(self):
        self.collections,self.writes = {},0
        self.ack_timeout = True
        self.optimize = True
    def handle(self,request):
        pieces = request.url.path.strip("/").split("/")
        assert pieces[0] == "collections"
        name = pieces[1]
        body = json.loads(request.content) if request.content else {}
        points = self.collections.get(name,{}).get("points",{})
        if len(pieces) == 2:
            if request.method == "GET":
                if name not in self.collections:
                    return httpx.Response(404,json={"status":"not found"})
                result = {"config":{"params":{"vectors":self.collections[name]["vectors"]}},
                    "status":"green" if self.optimize else "yellow","optimizer_status":"ok","indexed_vectors_count":len(points) if self.optimize else 0}
            else:
                assert request.method == "PUT"
                self.collections[name] = body | {"points":{}}
                result = True
        elif pieces[2:] == ["index"]:
            result = True
        elif pieces[2:] == ["points","count"]:
            filters = body.get("filter",{}).get("must",[])
            selected = [p for p in points.values() if all(p["payload"].get(f["key"]) == f["match"]["value"] for f in filters)]
            result = {"count":len(selected)}
        elif pieces[2:] == ["points"]:
            assert request.url.params["wait"] == "true"
            batch = body["batch"]
            for i,v,p in zip(batch["ids"],batch["vectors"],batch["payloads"],strict=True):
                points[i] = {"id":i,"vector":v,"payload":p}
            self.writes += 1
            if self.ack_timeout:
                self.ack_timeout = False
                raise httpx.ReadTimeout("lost acknowledgement",request=request)
            result = {"status":"completed"}
        elif pieces[2:] == ["points","query","batch"]:
            result = []
            for q in body["searches"]:
                assert q["params"]["exact"] is False
                ranked = sorted((float(np.dot(p["vector"],q["query"])),i,p) for i,p in points.items())[::-1][:q["limit"]]
                result.append({"points":[{"id":i,"score":s,"payload":p["payload"]} for s,i,p in ranked]})
        else:
            raise AssertionError(f"Unexpected Qdrant operation: {request.method} {request.url.path}")
        return httpx.Response(200,json={"status":"ok","result":result})


def test_qdrant_idempotent_retry_resume_append_and_missing_collection(tmp_path,monkeypatch):
    from vietmedbridge import qdrant_vectors as module
    monkeypatch.setenv("VMB_QDRANT_URL","http://localhost:6333")
    monkeypatch.setattr(module.time,"sleep",lambda _:None)
    server = Server()
    client = httpx.Client(base_url="http://localhost:6333",transport=httpx.MockTransport(server.handle))
    data = normalized(2048,8)
    store = Store(data[:1536])
    index = QdrantVectorIndex(store,tmp_path,policy(OPTIONS | {"backend":"qdrant"}),client=client)
    manifest = index.prepare()
    assert manifest["rows"] == 1536 and sum(s["rows"] for s in manifest["shards"]) == 1536
    ss,ii = index.search(data[-2:],20,128)
    truth = np.argsort(-(data[-2:] @ data[:1536].T),axis=1)[:,:20]
    np.testing.assert_array_equal(ii,truth)
    writes = server.writes
    store.calls.clear()
    index.prepare()
    assert server.writes == writes and not store.calls
    newer = Store(data)
    appended = QdrantVectorIndex(newer,tmp_path,index.config,client=client)
    appended.prepare()
    assert newer.calls == [6,7] and len(server.collections) == 4
    server.collections.pop(manifest["shards"][0]["collection"])
    store.calls.clear()
    index.prepare()
    assert store.calls == [0,1]  # server state, not local markers, determines coverage
    client.close()


def test_qdrant_unoptimized_is_not_ann_ready(tmp_path,monkeypatch):
    monkeypatch.setenv("VMB_QDRANT_URL","http://localhost:6333")
    server = Server();server.ack_timeout = False;server.optimize = False
    with httpx.Client(base_url="http://localhost:6333",transport=httpx.MockTransport(server.handle)) as client:
        index = QdrantVectorIndex(Store(normalized(11000,8),11000),tmp_path,
            policy({"backend":"qdrant"}),client=client)
        with pytest.raises(RuntimeError,match="WAITING_FOR_QDRANT_OPTIMIZATION"):
            index.prepare()
        assert read_json(tmp_path / "qdrant_progress.json")["state"] == "WAITING_FOR_QDRANT_OPTIMIZATION"


def test_qdrant_dispatcher_recall_refinement_and_resume(tmp_path,monkeypatch):
    from vietmedbridge import qdrant_vectors as module
    monkeypatch.setenv("VMB_QDRANT_URL","http://localhost:6333")
    server = Server();server.ack_timeout = False
    real_client = httpx.Client
    monkeypatch.setattr(module.httpx,"Client",lambda **kw:real_client(**kw,transport=httpx.MockTransport(server.handle)))
    store = Store(normalized(128,8),part_size=32)
    query = normalized(3,8,15)
    kwargs = {"k":16,"options":OPTIONS | {"backend":"qdrant"},"device":"cpu"}
    scores,ids,report = search_vectors(store,query,tmp_path,**kwargs)
    assert report["backend"] == "qdrant-server-hnsw-v1" and report["quality"]["recall"] == 1.
    truth = np.argsort(-(query @ store.values.T),axis=1)[:,:16]
    np.testing.assert_array_equal(ids,truth)
    writes = server.writes
    repeated = search_vectors(store,query,tmp_path,**kwargs)
    assert repeated[2]["manifest_sha256"] == report["manifest_sha256"] and server.writes == writes


def test_qdrant_rejects_secret_in_url_and_insecure_remote_key(tmp_path,monkeypatch):
    store = Store(normalized(2,8))
    monkeypatch.setenv("VMB_QDRANT_URL","http://user:secret@example.com:6333")
    with pytest.raises(ValueError,match="without credentials"):
        QdrantVectorIndex(store,tmp_path,policy())
    monkeypatch.setenv("VMB_QDRANT_URL","http://example.com:6333")
    monkeypatch.setenv("VMB_QDRANT_API_KEY","not-a-real-key")
    with pytest.raises(ValueError,match="HTTPS"):
        QdrantVectorIndex(store,tmp_path,policy())
