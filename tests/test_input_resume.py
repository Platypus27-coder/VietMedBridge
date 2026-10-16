"""Resume skips old payload reads; native readers retain SHA checks on use."""
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_scale_baseline import Tokenizer, Encoder
from vietmedbridge import index_inputs, input_resume, input_spool
from vietmedbridge.artifacts import atomic_json, digest_json, read_json, sha256_file
from vietmedbridge.content_embeddings import content_embeddings
from vietmedbridge.representation import representation_hash


@pytest.fixture
def frozen(tmp_path):
    """Twelve independent texts; the standard medical fixture deduplicates to two."""
    root = tmp_path / "synthetic-source"
    root.mkdir()
    spec = {"model_id":"SYNTHETIC_RESUME_TEST_ONLY","revision":"not-a-model"}
    documents,children = root / "documents.parquet",root / "children.parquet"
    pq.write_table(pa.Table.from_pylist([{"doc_id":1,"title":""}]),documents)
    rows = [{"retrieval_representation_hash":representation_hash(f"Original source text {n}"),
             "chunk_id":f"child-{n:03d}","doc_id":1,"text":f"Original source text {n}","heading":""} for n in range(12)]
    pq.write_table(pa.Table.from_pylist(rows),children)
    candidate = {"state":"FROZEN_CANDIDATE","selected_range_complete":True,
        "golden":{"tokenizer":spec},"chunking":{"dense_max_tokens":512},
        "counts":{"documents":1,"children":len(rows)},
        "health":{"chunks":{"distinct_representations":len(rows)}},
        "parts":[{"files":{"documents":{"path":documents.name,"sha256":sha256_file(documents)},
                            "children":{"path":children.name,"sha256":sha256_file(children)}}}]}
    candidate["candidate_manifest_sha256"] = digest_json(candidate)
    filename = f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json"
    atomic_json(root / "config.json",{"tokenizer":spec})
    atomic_json(root / filename,candidate)
    inputs = index_inputs.prepare_index_inputs(root,filename,Tokenizer(),part_size=1,work_dir=tmp_path / "original-work")
    return root,candidate,inputs


def prefix_at(tmp_path, frozen, selected=(0,1)):
    build,candidate,old = frozen
    assert len(old["parts"]) >= 3
    target = tmp_path / "resume-inputs"
    target.mkdir()
    shutil.copyfile(build / "index_inputs/config.json",target / "config.json")
    for index in selected:
        part = old["parts"][index]
        for name in (part["path"],f"units-{part['start']:012d}.done.json"):
            shutil.copyfile(build / "index_inputs" / name,target / name)
    args = (build,f"candidate-{candidate['candidate_manifest_sha256'][:16]}.json",Tokenizer())
    kwargs = {"work_dir":tmp_path / "resume-work","checkpoint_root":target,"part_size":1,
              "resume_policy":"verify_on_use"}
    return target,old,args,kwargs


def seed_cursor(target, old, count=2):
    parts = old["parts"][:count]
    last = pq.read_table(target / parts[-1]["path"],columns=["id"]).column("id").to_pylist()[-1]
    input_resume.InputResume(target,old["signature"],1,expected_total=old["input_count"]).save(parts,last,old["input_count"])
    return last


def test_legacy_bootstrap_reads_only_boundary_part_and_builds_tail(tmp_path,frozen,monkeypatch,capsys):
    target,old,args,kwargs = prefix_at(tmp_path,frozen)
    prefix = {target / part["path"] for part in old["parts"][:2]}
    boundary = target / old["parts"][1]["path"]
    real_verify,real_read = input_resume.verify_file,pq.read_table
    reads,counts = [],[]
    def verify(path,sha):
        if Path(path) in prefix:
            assert Path(path) == boundary
            reads.append(Path(path))
        return real_verify(path,sha)
    def read(path,*a,**k):
        if Path(path) in prefix:
            assert Path(path) == boundary
        return real_read(path,*a,**k)
    real_stage = input_spool.InputSpool.stage
    def stage(self):
        count = real_stage(self)
        counts.append((self.minimum_id,count))
        return count
    monkeypatch.setattr(input_resume,"verify_file",verify)
    monkeypatch.setattr(index_inputs,"verify_file",verify)
    monkeypatch.setattr(pq,"read_table",read)
    monkeypatch.setattr(input_spool.InputSpool,"stage",stage)
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    assert reads == [boundary]
    assert counts[0][0] is not None and counts[0][1] == old["input_count"]-2
    assert result["parts"] == old["parts"]
    assert "Index inputs RESUME: 2/" in capsys.readouterr().out


def test_sealed_cursor_reads_no_old_payload_before_continuing(tmp_path,frozen,monkeypatch):
    target,old,args,kwargs = prefix_at(tmp_path,frozen)
    seed_cursor(target,old)
    old_paths = {target / p["path"] for p in old["parts"][:2]}
    original = pq.read_table
    def read(path,*a,**k):
        assert Path(path) not in old_paths,"Cursor resume reread old payload"
        return original(path,*a,**k)
    monkeypatch.setattr(pq,"read_table",read)
    monkeypatch.setattr(input_resume,"verify_file",lambda *a:pytest.fail("Cursor resume hashed old payload"))
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    assert result["parts"] == old["parts"] and result["input_count"] == old["input_count"]


@pytest.mark.parametrize("fresh_runtime",[False,True])
def test_interrupted_cursor_resumes_missing_part_and_keeps_previous_bytes(tmp_path,frozen,monkeypatch,fresh_runtime):
    target,old,args,kwargs = prefix_at(tmp_path,frozen,selected=(0,))
    original = index_inputs.atomic_json
    def interrupt(path,value):
        if Path(path).name == "units-000000000002.done.json":
            raise OSError("interrupted third part")
        return original(path,value)
    monkeypatch.setattr(index_inputs,"atomic_json",interrupt)
    with pytest.raises(OSError,match="third part"):
        index_inputs.prepare_index_inputs(*args,**kwargs)
    saved = read_json(target / "resume.json")
    assert saved["cursor"] == 2 and saved["spool_base_rows"] == 1
    first = target / old["parts"][0]["path"]
    before = sha256_file(first),first.stat().st_mtime_ns
    if fresh_runtime:
        shutil.rmtree(kwargs["work_dir"])
    monkeypatch.setattr(index_inputs,"atomic_json",original)
    monkeypatch.setattr(input_resume,"verify_file",lambda *a:pytest.fail("Saved cursor reread prefix"))
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    assert result["parts"] == old["parts"]
    assert (sha256_file(first),first.stat().st_mtime_ns) == before


def test_noncontiguous_receipts_fill_gap_without_dropping_later_part(tmp_path,frozen):
    target,old,args,kwargs = prefix_at(tmp_path,frozen,selected=(0,2))
    later = target / old["parts"][2]["path"]
    before = sha256_file(later),later.stat().st_mtime_ns
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    assert result["parts"] == old["parts"]
    assert (sha256_file(later),later.stat().st_mtime_ns) == before


def test_wrong_cursor_contract_and_missing_payload_are_rejected(tmp_path,frozen):
    target,old,args,kwargs = prefix_at(tmp_path,frozen)
    seed_cursor(target,old)
    path = target / "resume.json"
    saved = read_json(path)
    changed = saved | {"signature":"f"*64}
    changed["manifest_sha256"] = digest_json({k:v for k,v in changed.items() if k != "manifest_sha256"})
    atomic_json(path,changed)
    with pytest.raises(ValueError,match="cursor changed"):
        index_inputs.prepare_index_inputs(*args,**kwargs)
    atomic_json(path,saved)
    (target / old["parts"][0]["path"]).unlink()
    with pytest.raises(ValueError,match="Missing index input"):
        index_inputs.prepare_index_inputs(*args,**kwargs)


def test_deferred_corruption_stops_native_consumer_before_encoding(tmp_path,frozen):
    target,old,args,kwargs = prefix_at(tmp_path,frozen)
    seed_cursor(target,old)
    first = target / old["parts"][0]["path"]
    first.write_bytes(b"damaged after publication")
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    encoder = Encoder()
    with pytest.raises(ValueError,match="changed artifact"):
        content_embeddings(tmp_path,target,result,encoder,tmp_path / "vectors",work_dir=tmp_path / "vector-work")
    assert encoder.calls == 0


def test_completed_cursor_resume_avoids_spool_and_old_payload_scan(tmp_path,frozen,monkeypatch):
    target,old,args,kwargs = prefix_at(tmp_path,frozen)
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    monkeypatch.setattr(input_spool.InputSpool,"stage",lambda *a:pytest.fail("Completed inputs rebuilt spool"))
    monkeypatch.setattr(index_inputs,"verify_file",lambda *a:pytest.fail("Complete manifest rehashed old payloads"))
    assert index_inputs.prepare_index_inputs(*args,**kwargs) == result


def test_all_parts_saved_without_manifest_skip_spool(tmp_path,frozen,monkeypatch):
    target,old,args,kwargs = prefix_at(tmp_path,frozen,selected=tuple(range(len(frozen[2]["parts"]))))
    seed_cursor(target,old,count=len(old["parts"]))
    monkeypatch.setattr(input_spool.InputSpool,"stage",lambda *a:pytest.fail("All saved parts rebuilt spool"))
    monkeypatch.setattr(input_resume,"verify_file",lambda *a:pytest.fail("Sealed prefix reread payload"))
    result = index_inputs.prepare_index_inputs(*args,**kwargs)
    assert result["state"] == "COMPLETE" and result["parts"] == old["parts"]
