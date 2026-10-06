"""BGE document retrieval representations: source title plus bounded opening."""
from pathlib import Path

from .embeddings import embed_units, embedding_matrix
from .quality import error_page_reason, has_encoded_payload
from .retrieval_cache import reuse_bge_cache
from .text import _redirected_to_homepage


def document_units(catalog, tokenizer, max_length=512):
    limit = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if limit < 32:
        raise ValueError("Document embedding budget is too small.")
    ids = {c["doc_id"] for c in catalog.children.values()}
    units = []
    for identifier in sorted(ids):
        doc = catalog.documents[identifier]
        source = doc["source_text"]
        if (error_page_reason(doc.get("title", ""), source) or has_encoded_payload(source)
            or (doc.get("url") and doc.get("final_url") and _redirected_to_homepage(doc["url"], doc["final_url"]))):
            continue
        title = doc.get("title", "").strip()
        offsets = tokenizer(title, add_special_tokens=False, return_offsets_mapping=True,truncation=False,verbose=False)["offset_mapping"]
        if len(offsets) > 64:
            title = title[:offsets[63][1]]
        title_cost = len(tokenizer(title, add_special_tokens=False,truncation=False,verbose=False)["input_ids"])
        offsets = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True,truncation=False,verbose=False)["offset_mapping"]
        count = min(len(offsets), limit - title_cost - 4)
        while count > 0:
            excerpt = source[:offsets[count - 1][1]]
            text = title + "\n\n" + excerpt if title else excerpt
            if len(tokenizer(text, add_special_tokens=False,truncation=False,verbose=False)["input_ids"]) <= limit:
                break
            count -= 1
        else:
            raise ValueError("Document opening cannot fit embedding budget.")
        units.append({"id": str(identifier), "text": text})
    if not units:
        raise ValueError("No document dense representations after quarantine.")
    return units


def prepare_document_vectors(catalog, tokenizer, spec, output_dir, encoder_factory, *, embedding, work_dir=None):
    units = document_units(catalog, tokenizer, spec["max_length"])
    cache = reuse_bge_cache(output_dir, units, spec, part_size=embedding["part_size"])
    if cache is not None:
        vectors, manifest = cache
    else:
        encoder = encoder_factory(spec)
        try:
            manifest = embed_units(units, encoder, output_dir, work_dir=work_dir, **embedding)
        finally:
            encoder.close()
        vectors = embedding_matrix(Path(output_dir), manifest)
    return vectors, manifest, units
