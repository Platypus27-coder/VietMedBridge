"""Verified legacy mining replay and cheap multi-channel training candidates."""
from dataclasses import asdict
import math
from pathlib import Path

from .artifacts import atomic_json, digest_json, read_json, sha256_file, verify_file


def cached_mining_records(run, queries, catalog, config):
    """Load completed legacy mining before loading a GPU model or rebuilding index."""
    run = Path(run)
    paths = [run / "mining_queries" / f"query-{q['id']}.done.json" for q in queries]
    contract_path, index_path = run / "mining_queries/config.json", run / "index/strong.json"
    if not contract_path.is_file() or not index_path.is_file() or not all(p.is_file() for p in paths):
        directory = run / "retrieval_mining"
        paths = [directory / f"query-{q['id']}.done.json" for q in queries]
        if not index_path.is_file() or not (directory / "config.json").is_file() or not all(p.is_file() for p in paths):
            return None
        contract, index = read_json(directory / "config.json"), read_json(index_path)
        if (digest_json({k:v for k,v in index.items() if k != "manifest_sha256"}) != index["manifest_sha256"]
            or contract != {"index": index["signature"], "queries": queries, "config": asdict(config),
                            "mode": "MULTICHANNEL_RETRIEVAL_NO_RERANK"} or index["catalog"] != catalog.identity):
            raise ValueError("Cached retrieval mining source/query/policy mismatch.")
        records = [read_json(p) for p in paths]
        for query, record in zip(queries, records, strict=True):
            if (record["signature"] != digest_json(contract) or record["query_sha256"] != digest_json(query)
                or record["prediction"]["id"] != query["id"]
                or digest_json({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"]
                or any(catalog.children[r["child_id"]]["doc_id"] != r["doc_id"] for r in record["ranking"]["children"])):
                raise ValueError("Cached retrieval mining query integrity mismatch.")
        return records
    contract, index = read_json(contract_path), read_json(index_path)
    if (digest_json({k: v for k, v in contract.items() if k != "signature"}) != contract["signature"]
        or digest_json({k: v for k, v in index.items() if k != "manifest_sha256"}) != index["manifest_sha256"]):
        raise ValueError("Cached mining/index contract integrity mismatch.")
    if (index["catalog"] != catalog.identity or contract["index"] != index["signature"]
        or contract["queries_sha256"] != digest_json(queries) or contract["config"] != asdict(config)):
        raise ValueError("Cached mining source/query/policy changed.")
    for key, name in (("code_sha256", "strong_retrieval.py"), ("selection_code_sha256", "retrieval_policy.py")):
        if contract.get(key) != sha256_file(Path(__file__).with_name(name)):
            raise ValueError("Cached mining inference/selection code changed.")
    records = []
    for query, path in zip(queries, paths, strict=True):
        record = read_json(path)
        if (record["signature"] != contract["signature"] or record["query_sha256"] != digest_json(query)
            or record["prediction"]["id"] != query["id"]
            or digest_json({k: v for k, v in record.items() if k != "record_sha256"}) != record["record_sha256"]):
            raise ValueError("Cached mining query integrity mismatch.")
        for stage, checksum in record["score_checkpoints"].items():
            if stage not in ("documents", "children"):
                raise ValueError("Unknown cached mining score stage.")
            evidence = read_json(path.parent / f"query-{query['id']}-{stage}.done.json")
            if (evidence["record_sha256"] != checksum
                or digest_json({k: v for k, v in evidence.items() if k != "record_sha256"}) != checksum
                or evidence["signature"] != contract["signature"] or evidence["reranker"] != contract["reranker"]
                or evidence["pairs_sha256"] != digest_json(evidence["pairs"])
                or len(evidence["pairs"]) != len(evidence["scores"])
                or not all(math.isfinite(s) for s in evidence["scores"])):
                raise ValueError("Cached mining scored-stage integrity mismatch.")
        if any(catalog.children[r["child_id"]]["doc_id"] != r["doc_id"] for r in record["ranking"]["children"]):
            raise ValueError("Cached mining child/source mismatch.")
        records.append(record)
    return records


def retrieve_mining_candidates(index, queries, vectors, translations, config, output_dir):
    """Keep wide retrieval; reserve expensive 8B reranking for training/dev inference.

    These ranks are candidate search aids, never inferred positive/negative labels.
    Unknown passages remain unlabelled, and missing source positives cause holds.
    """
    root = Path(output_dir)
    contract = {"index": index.manifest["signature"], "queries": queries,
                "config": asdict(config), "mode": "MULTICHANNEL_RETRIEVAL_NO_RERANK"}
    signature = digest_json(contract)
    path = root / "config.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("Retrieval mining inputs changed.")
    atomic_json(path, contract)
    records = []
    for query, vector, translation in zip(queries, vectors, translations, strict=True):
        path = root / f"query-{query['id']}.done.json"
        if path.is_file():
            record = read_json(path)
            if (record["signature"] != signature
                or digest_json({k: v for k, v in record.items() if k != "record_sha256"}) != record["record_sha256"]):
                raise ValueError("Retrieval mining checkpoint integrity mismatch.")
        else:
            documents, scores = index.document_candidates(query["query"], vector, translation["variants"], config)
            children = []
            for doc_rank, doc in enumerate(documents, 1):
                hits = index.child_candidates(doc["doc_id"], query["query"], translation["variants"], scores, config, config.children_per_doc)
                for child_rank, hit in enumerate(hits, 1):
                    children.append(hit | {"mining_rrf": 1 / (config.rrf_k + doc_rank) + 1 / (config.rrf_k + child_rank)})
            children.sort(key=lambda r: (-r["mining_rrf"], r["child_id"]))
            record = {"signature": signature, "query_sha256": digest_json(query), "prediction": {"id": query["id"]},
                "ranking": {"documents": documents, "children": children}, "scope": contract["mode"]}
            record["record_sha256"] = digest_json(record)
            atomic_json(path, record)
        records.append(record)
    return records


def unique_finalists(paths):
    """Final adapter often duplicates the last epoch checkpoint: evaluate it once."""
    seen, result = set(), []
    for path in paths:
        manifest = read_json(Path(path) / "adapter_manifest.json")
        if digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
            raise ValueError("Finalist manifest integrity mismatch.")
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            verify_file(Path(path) / name, manifest["files"][name])
        key = (manifest["base_model"], manifest["base_revision"], manifest["files"]["adapter_model.safetensors"], manifest["files"]["adapter_config.json"])
        if key not in seen:
            result.append(Path(path))
            seen.add(key)
    return result
