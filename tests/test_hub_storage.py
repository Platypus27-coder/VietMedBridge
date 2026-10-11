"""Real snapshot/catalog/ZIP wiring; Hub API and GPU models are explicit doubles."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_external_import import external, SPEC  # noqa: F401
from test_scale_baseline import Encoder, Tokenizer, frozen, build_source  # noqa: F401
from vietmedbridge.artifacts import atomic_json, read_json, sha256_file
from vietmedbridge.catalog_storage import prepare_disk_catalog
from vietmedbridge.competition_pilot import MASTER_PLAN
from vietmedbridge.content_embeddings import content_embeddings
from vietmedbridge import hub_storage as storage
from vietmedbridge.medical_lexical import MedicalAnalyzer
from vietmedbridge.portable_embeddings import resources_relative

REPO = Path(__file__).resolve().parents[1]


class DenseDouble(Encoder):
    dimension = 1024
    def __init__(self,spec):
        super().__init__()
        self.identity.update(spec)
        self.texts = []
    def encode(self,texts,**kwargs):
        self.texts.extend(texts)
        return np.pad(super().encode(texts,**kwargs),((0,0),(0,1020)))


class MemoryHub:
    token = None
    def __init__(self,tmp_path):
        self.root,self.files,self.history = tmp_path,{},{}
        self.private,self.commits,self.fail_commit = True,0,None
        self.sha = "0" * 40
    def create_repo(self,*args,**kwargs):
        assert kwargs["private"] is True and kwargs["repo_type"] == "dataset"
    def dataset_info(self,*args,revision=None,**kwargs):
        if revision and revision != "main" and revision not in self.history:
            raise ValueError("Unknown revision")
        return SimpleNamespace(private=self.private,sha=revision if revision and revision != "main" else self.sha)
    def get_paths_info(self,repo,paths,**kwargs):
        result = []
        for path in paths:
            if path in self.files:
                # Exercise both server-side LFS verification and ordinary Git blobs.
                data = self.files[path]
                lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if len(data) > 2000 else None
                result.append(SimpleNamespace(path=path,size=len(data),lfs=lfs))
        return result
    def _commit(self):
        self.commits += 1
        self.sha = f"{self.commits:040x}"
        self.history[self.sha] = deepcopy(self.files)
        return SimpleNamespace(oid=self.sha)
    def create_commit(self,repo,*,operations,**kwargs):
        if self.commits + 1 == self.fail_commit:
            raise ConnectionError("Simulated network interruption")
        for operation in operations:
            self.files[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        return self._commit()
    def upload_file(self,*,path_in_repo,path_or_fileobj,**kwargs):
        self.files[path_in_repo] = path_or_fileobj
        return self._commit()
    def download(self,repo,filename,revision,cache,api):
        contents = self.files if revision == "main" else self.history[revision]
        path = self.root / ("dl-"+hashlib.sha256((revision+filename).encode()).hexdigest()[:24])
        path.write_bytes(contents[filename])
        return path


@pytest.fixture
def prepared(tmp_path,frozen,external):
    build,candidate,inputs = frozen
    checkout = tmp_path / "checkout"
    config = read_json(REPO / "configs/retrieval_large_baseline.json")
    config.update(dense=SPEC,lexical_segmentation=False)
    full = {"dense":SPEC,"second_dense":{"model_id":"unused-test-Qwen"},"lexical_segmentation":False,
        "pilot_limits":{"max_documents":2000,"max_children":50000}}
    atomic_json(checkout / "configs/retrieval_large_baseline.json",config)
    atomic_json(checkout / "configs/retrieval_full.json",full)
    atomic_json(checkout / "configs/strong_model_manifest.json",{"models":[SPEC | {"parameters":4}]})
    (checkout / MASTER_PLAN).write_text("CPU test fixture; not competition predictions",encoding="utf-8")
    queries = [{"id":10001+i*7,"query":"Người bệnh triệu chứng 中文医学"} for i in range(1200)]
    query = tmp_path / "query.parquet"
    pq.write_table(pa.Table.from_pylist(queries),query)
    atomic_json(tmp_path / "raw/snapshot.json",{"files":{
        "query.parquet":{"path":"query.parquet","sha256":sha256_file(query)},
        "links_corpus.parquet":{"path":external[1].relative_to(tmp_path).as_posix(),"sha256":candidate["origin_corpus_sha256"]}}})
    catalog = prepare_disk_catalog(build,candidate,inputs,tmp_path / "retrieval/catalogs",
        MedicalAnalyzer(segmentation=False),work_dir=tmp_path / "work")
    catalog.close()
    encoder = DenseDouble(SPEC)
    view = tmp_path / resources_relative(candidate,full) / "bge"
    content_embeddings(tmp_path,build / "index_inputs",inputs,encoder,view,work_dir=tmp_path / "work")
    qwen = DenseDouble(full["second_dense"])
    content_embeddings(tmp_path,build / "index_inputs",inputs,qwen,view.parent / "qwen",
        work_dir=tmp_path / "work",max_new_parts=1)
    plan = storage.plan_active_storage(tmp_path,checkout)
    encoder.texts.clear()
    return checkout,plan,encoder,queries


@pytest.fixture
def hub(tmp_path,monkeypatch):
    api = MemoryHub(tmp_path)
    monkeypatch.setattr(storage,"_download",api.download)
    return api


def test_interrupted_migration_resumes_after_local_state_loss_and_never_writes_source(tmp_path,prepared,hub):
    checkout,plan,_,_ = prepared
    before = {i["path"]:(sha256_file(tmp_path / i["path"]),(tmp_path / i["path"]).stat().st_mtime_ns) for i in plan["files"]}
    work = tmp_path.parent / "upload"
    # Work intentionally stays outside the source root, as in /content/vmb_storage.
    hub.fail_commit = 2
    with pytest.raises(ConnectionError,match="interruption"):
        storage.upload_storage(tmp_path,plan,work_dir=work,api=hub)
    assert len(hub.files) == 20 and "vmb-latest.json" not in hub.files
    hub.fail_commit = None
    result = storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "retry",api=hub)
    assert result["state"] == "HF_STORAGE_VERIFIED" and result["reused_objects"] == 20
    assert result["source_deleted"] is False and result["source_writes"] is False
    again = storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "third",api=hub)
    assert again["new_objects"] == 0 and again["reused_objects"] == storage.storage_summary(plan)["unique_objects"]
    assert {i["path"]:(sha256_file(tmp_path / i["path"]),(tmp_path / i["path"]).stat().st_mtime_ns) for i in plan["files"]} == before


def test_budget_and_public_repo_stop_before_payload_upload(tmp_path,prepared,hub):
    _,plan,_,_ = prepared
    work = tmp_path.parent / "upload"
    result = storage.upload_storage(tmp_path,plan,work_dir=work,max_snapshot_bytes=1,api=hub)
    assert result["state"] == "UPLOAD_NOT_STARTED_BUDGET_EXCEEDED" and not hub.files
    hub.private = False
    with pytest.raises(ValueError,match="PRIVATE"):
        storage.upload_storage(tmp_path,plan,work_dir=work,api=hub)
    assert not hub.files


def test_changed_payload_never_publishes_complete_pointer(tmp_path,prepared,hub):
    _,plan,_,_ = prepared
    changed = next(i for i in plan["files"] if i["path"].endswith(".npy"))
    (tmp_path / changed["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="changed artifact"):
        storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "upload",api=hub)
    assert "vmb-latest.json" not in hub.files


def test_snapshot_restore_and_full_1200_query_zip_without_drive_writes_or_corpus_encoding(tmp_path,prepared,hub,monkeypatch):
    from vietmedbridge import scale_baseline as runtime
    checkout,plan,encoder,queries = prepared
    plan = storage.plan_active_storage(tmp_path,checkout,profile="bge_submit")
    assert plan["snapshot_profile"] == "bge_submit" and not plan["includes_source_shards"]
    assert plan["coverage"]["qwen"] == {"included":False}
    assert all(set(i["roles"]) <= {"metadata","catalog","bge"} for i in plan["files"])
    result = storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "upload",api=hub)
    destination = tmp_path.parent / (tmp_path.name+"-runtime-data")
    local_work = tmp_path.parent / (tmp_path.name+"-gpu-work")
    staged = storage.restore_storage(data_root=destination,work_dir=local_work,
        revision=result["revision"],api=hub,checkout=checkout)
    assert staged["state"] == "HF_DATA_READY_LOCAL"
    assert not list(destination.glob("processed/*/part-*/documents.parquet"))
    assert not any("qwen" in p.parts for p in destination.rglob("map-*.done.json"))
    repeated = storage.restore_storage(data_root=destination,work_dir=local_work,
        revision=result["revision"],api=hub,checkout=checkout)
    assert repeated == staged
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)))
    monkeypatch.setattr(runtime,"review_model_registry",lambda *_:None)
    monkeypatch.setattr(runtime,"TorchDenseEncoder",lambda _:encoder)
    monkeypatch.setattr(runtime,"stream_embeddings",lambda *a,**kw:pytest.fail("Corpus reencoding forbidden"))
    real_search = runtime.search_parts
    monkeypatch.setattr(runtime,"search_parts",lambda *a,**kw:real_search(*a,**kw,device="cpu"))
    # The full source is in the restored SQLite catalog; processed source shards
    # and input text Parquets are deliberately absent in bge_submit mode.
    output = runtime.run_large_baseline(destination,checkout,code_commit="CPU-double",
        work_dir=local_work,reuse_content_only=True)
    assert output["ready"]["state"] == "READY_FOR_MANUAL_UPLOAD"
    assert encoder.texts == [q["query"] for q in queries]
    with ZipFile(output["ready"]["zip_path"]) as archive:
        predictions = json.loads(archive.read("results.json"))
    assert [p["id"] for p in predictions] == [q["id"] for q in queries]
    sources = {d:build_source(tmp_path,d) for d in (583,921)}
    assert all(c["chunk_text"] in sources[c["doc_id"]] for p in predictions for c in p["relevant_chunks"])
    assert Path(output["ready"]["zip_path"]).is_relative_to(destination)


def test_incomplete_bge_snapshot_cannot_be_restored_for_submit(tmp_path,prepared,hub):
    _,plan,_,_ = prepared
    from vietmedbridge.content_embeddings import seal
    incomplete = deepcopy(plan)
    incomplete.pop("manifest_sha256")
    incomplete["coverage"]["bge"]["complete"] = False
    incomplete = seal(incomplete)
    result = storage.upload_storage(tmp_path,incomplete,work_dir=tmp_path.parent / "upload",api=hub)
    with pytest.raises(ValueError,match="complete BGE"):
        storage.restore_storage(data_root=tmp_path.parent / (tmp_path.name+"-runtime-data"),work_dir=tmp_path.parent / "work",
            revision=result["revision"],api=hub)


def test_full_profile_restores_every_planned_file(tmp_path,prepared,hub):
    checkout,plan,_,_ = prepared
    result = storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "upload",api=hub)
    destination = tmp_path.parent / (tmp_path.name+"-full-data")
    restored = storage.restore_storage(data_root=destination,work_dir=tmp_path.parent / "work",
        profile="all",revision=result["revision"],api=hub,checkout=checkout)
    assert restored["state"] == "HF_DATA_READY_LOCAL"
    assert restored["restored_files"] == len(plan["files"])
    assert all(sha256_file(destination / item["path"]) == item["sha256"] for item in plan["files"])


def test_submission_snapshot_ignores_qwen_and_does_not_claim_full_pipeline(tmp_path,prepared,hub):
    checkout,plan,_,_ = prepared
    _,candidate,_ = storage.load_handoff(tmp_path)
    full = read_json(checkout / "configs/retrieval_full.json")
    qwen = tmp_path / resources_relative(candidate,full) / "qwen/config.json"
    qwen.write_text("corrupted unrelated Qwen checkpoint",encoding="utf-8")
    fast = storage.plan_active_storage(tmp_path,checkout,profile="bge_submit")
    assert len(fast["files"]) < len(plan["files"])
    result = storage.upload_storage(tmp_path,fast,work_dir=tmp_path.parent / "upload",api=hub)
    with pytest.raises(ValueError,match="BGE submission snapshot"):
        storage.restore_storage(data_root=tmp_path.parent / (tmp_path.name+"-full-data"),
            work_dir=tmp_path.parent / "work",profile="all",revision=result["revision"],api=hub)
    bge_marker = tmp_path / next(i["path"] for i in fast["files"]
        if "bge" in i["roles"] and Path(i["path"]).name.startswith("map-") and i["path"].endswith(".done.json"))
    bge_marker.unlink()
    with pytest.raises(ValueError,match="requires complete BGE"):
        storage.plan_active_storage(tmp_path,checkout,profile="bge_submit")


def test_catalog_mismatch_stops_before_downloading_vectors(tmp_path,prepared,hub,monkeypatch):
    from vietmedbridge import catalog_storage
    checkout,plan,_,_ = prepared
    result = storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "upload",api=hub)
    downloaded = []
    def observed(*args):
        downloaded.append(args[1])
        return hub.download(*args)
    monkeypatch.setattr(storage,"_download",observed)
    original = catalog_storage._contract
    monkeypatch.setattr(catalog_storage,"_contract",lambda *a:original(*a) | {"sqlite":"different-runtime"})
    destination = tmp_path.parent / (tmp_path.name+"-runtime-data")
    with pytest.raises(ValueError,match="Saved CPU catalog differs"):
        storage.restore_storage(data_root=destination,work_dir=tmp_path.parent / "work",
            revision=result["revision"],api=hub,checkout=checkout)
    vector_objects = {storage.object_path(i) for i in plan["files"] if i["path"].endswith(".npy")}
    assert not vector_objects.intersection(downloaded)
    assert read_json(destination / "hf_storage_binding.json")["state"] == "STAGING"


def test_corrupt_remote_payload_cannot_mark_restore_ready(tmp_path,prepared,hub):
    checkout,plan,_,_ = prepared
    result = storage.upload_storage(tmp_path,plan,work_dir=tmp_path.parent / "upload",api=hub)
    item = next(i for i in plan["files"] if i["path"].endswith(".npy") and "bge" in i["roles"])
    hub.history[result["revision"]][storage.object_path(item)] = b"corrupt"
    destination = tmp_path.parent / (tmp_path.name+"-runtime-data")
    with pytest.raises(ValueError,match="changed artifact"):
        storage.restore_storage(data_root=destination,work_dir=tmp_path.parent / "work",
            revision=result["revision"],api=hub,checkout=checkout)
    assert read_json(destination / "hf_storage_binding.json")["state"] == "STAGING"


def test_notebooks_are_valid_and_hf_profile_selects_cached_coordinator(tmp_path,prepared,monkeypatch):
    import nbformat
    from vietmedbridge import full_plan_runtime, full_scale_runtime, scale_baseline
    checkout,_,_,_ = prepared
    for path in (REPO / "notebooks/04_colab_retrieval_baseline.ipynb",REPO / "notebooks/tools/drive_hf_storage.ipynb"):
        notebook = nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source,str(path),"exec")
    notebook = read_json(REPO / "notebooks/04_colab_retrieval_baseline.ipynb")
    cell = next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and "PREPARE_ONLY =" in c["source"])
    calls = []
    def forbidden(*args,**kwargs):
        pytest.fail("HF submit must not enter full inference or CPU rebuild")
    monkeypatch.setattr(full_plan_runtime,"run_full_pipeline",forbidden)
    monkeypatch.setattr(full_scale_runtime,"run_scale_full_pipeline",forbidden)
    monkeypatch.setattr(full_scale_runtime,"prepare_scale_cpu_resources",forbidden)
    def cached(*args,**kwargs):
        calls.append(kwargs)
        return {"ready":{"state":"READY_FOR_MANUAL_UPLOAD","query_count":1200},"status":{},"samples":[]}
    monkeypatch.setattr(scale_baseline,"run_large_baseline",cached)
    context = {"DATA_ROOT":tmp_path,"CHECKOUT":checkout,"WORK_DIR":tmp_path.parent / "work",
        "CODE_COMMIT":"CPU-test","USE_HF_STORAGE":True,"HF_STORAGE_PROFILE":"bge_submit"}
    exec(compile(cell,"04-hf-routing","exec"),context)
    assert calls[0]["reuse_content_only"] is True
    assert context["TEAM_SIZE"] == 1 and context["TEAM_WORKER_ID"] is None and context["PREPARE_ONLY"] is False
