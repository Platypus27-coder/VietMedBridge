"""Local lexical throughput/parity probe; not a Colab or relevance benchmark."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import time

from vietmedbridge.catalog_builder import _analyze_document, _init_analyzer
from vietmedbridge.disk_catalog import lexical_field, packed
from vietmedbridge.medical_lexical import MedicalAnalyzer


def original(document,analyzer):
    fields = [document["title"],"\n".join(document["heading_hints"]),document["source_text"],analyzer.aliases(document["source_text"])]
    return [{"doc_id":document["doc_id"],"language":"vi",
        **dict(zip(("title","headings","body","aliases"),
                   [lexical_field(analyzer,t,"vi") for t in fields],strict=True))}]


def main():
    analyzer = MedicalAnalyzer()
    text = ("Người bệnh bị viêm dạ dày, đau thượng vị và khó nuốt. "
            "Cần theo dõi triệu chứng và đáp ứng điều trị. ")*80
    docs = [{"doc_id":i,"language":"vi","title":f"Viêm dạ dày {i}",
             "heading_hints":["Chẩn đoán và điều trị"],"source_text":text+str(i)} for i in range(512)]
    start = time.perf_counter()
    expected = [original(d,analyzer) for d in docs]
    serial = time.perf_counter()-start
    checksum = hashlib.sha256(packed(expected).encode()).hexdigest()
    timings = []
    for workers in (2,4):
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_analyzer,initargs=(analyzer.identity,analyzer.entries)) as pool:
            actual = list(pool.map(_analyze_document,docs,chunksize=4))
        elapsed = time.perf_counter()-start
        assert hashlib.sha256(packed(actual).encode()).hexdigest() == checksum
        timings.append({"workers":workers,"seconds":round(elapsed,2),"speedup":round(serial/elapsed,2)})
    print(json.dumps({"scope":"synthetic lexical CPU only; includes process startup; not Colab ETA",
        "documents":len(docs),"characters_per_document":len(text),"original_seconds":round(serial,2),
        "parallel":timings,"exact_terms_equal":True},indent=2),flush=True)


if __name__ == "__main__":
    main()
