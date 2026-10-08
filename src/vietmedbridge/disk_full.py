"""Full-plan candidate fusion and lazy adaptive parents over the disk catalog."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .artifacts import digest_json, sha256_file
from .disk_catalog import lexical_field
from .quality import error_page_reason, has_encoded_payload
from .representation import build_sparse_text
from .retrieval_cascade import PostingBM25, weighted_rrf
from .retrieval_data import Catalog
from .source_parents import derive_parents
from .text import _redirected_to_homepage


class TableRows(Mapping):
    def __init__(self,base,table,key):
        self.base,self.table,self.key = base,table,key
        self.rows = getattr(base,table)
    def __getitem__(self,key):
        return self.rows[key]
    def __contains__(self,key):
        return key in self.rows
    def __iter__(self):
        for row in self.base.con.execute(f"SELECT {self.key} FROM {self.table} ORDER BY {self.key}"):
            yield row[0]
    def __len__(self):
        return self.base.con.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]


class AdaptiveRows(Mapping):
    def __init__(self,catalog,kind):
        self.catalog,self.kind = catalog,kind
    def __getitem__(self,key):
        if self.kind == "children":
            child = self.catalog.base.children[key]
            return self.catalog.derived(child["doc_id"])[0][key]
        if key.startswith("disk-parent-"):
            try:
                doc = int(key.split("-",3)[2])
            except (ValueError,IndexError):
                raise KeyError(key)
            return self.catalog.derived(doc)[1][key]
        return self.catalog.base.parents[key]
    def __iter__(self):
        return iter(TableRows(self.catalog.base,self.kind,"chunk_id"))
    def __len__(self):
        return len(TableRows(self.catalog.base,self.kind,"chunk_id"))


class AdaptiveDiskCatalog:
    def __init__(self,base,tokenizer,budgets=(512,640)):
        self.base,self.tokenizer,self.budgets = base,tokenizer,tuple(budgets)
        self.documents = TableRows(base,"documents","doc_id")
        self.children,self.parents = AdaptiveRows(self,"children"),AdaptiveRows(self,"parents")
        self.candidate = base.candidate
        self.cache = OrderedDict()
        self.build_config = {"official_links_sha256":base.candidate["origin_corpus_sha256"]}
        self.parent_policy = {"budgets":list(budgets),"policy":"disk-lazy-center-anchor-v1",
            "tokenizer":base.candidate["golden"]["tokenizer"],
            "code_sha256":sha256_file(Path(__file__).with_name("source_parents.py"))}
    @property
    def identity(self):
        return self.base.identity | {"derived_parent_policy_sha256":digest_json(self.parent_policy)}
    def derived(self,doc):
        if doc not in self.cache:
            pairs = self.base.document_children(doc)
            source = {doc:self.documents[doc]}
            children = {r["chunk_id"]:r for r,_ in pairs}
            parents = {r["parent_id"]:self.base.parents[r["parent_id"]] for r,_ in pairs}
            small = Catalog(self.candidate,source,children,parents,[],{},self.build_config)
            value = derive_parents(small,self.tokenizer,self.budgets)
            renamed = {}
            for child in value.children.values():
                for budget,old in child["source_parent_alternatives"].items():
                    new = f"disk-parent-{doc}-"+old.removeprefix("source-parent-")
                    renamed[new] = value.parents[old] | {"chunk_id":new}
                    child["source_parent_alternatives"][budget] = new
            self.cache[doc] = value.children,renamed
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(doc)
        return self.cache[doc]
    def close(self):
        self.base.close()


class Languages(Mapping):
    def __init__(self,catalog):
        self.catalog = catalog
    def __getitem__(self,doc):
        return self.catalog.documents[doc].get("language","unknown")
    def __iter__(self):
        return iter(self.catalog.documents)
    def __len__(self):
        return len(self.catalog.documents)


class DiskStrongIndex:
    """Preserve the strong cascade API without materializing corpus vectors/text."""
    def __init__(self,catalog,primary,secondary,query_vectors,auxiliary,queries,expansions,
                 primary_search,secondary_search,document_search,document_ids,tokenizer,manifest):
        self.catalog,self.base,self.tokenizer = catalog,catalog.base,tokenizer
        self.primary,self.secondary,self.auxiliary = primary,secondary,auxiliary
        self.query_vectors = {q["query"]:v for q,v in zip(queries,query_vectors,strict=True)}
        self.expansions = {q["query"]:e["variants"] for q,e in zip(queries,expansions,strict=True)}
        self.primary_search,self.secondary_search,self.document_search = primary_search,secondary_search,document_search
        self.document_ids = document_ids
        self.encoder = primary.manifest["encoder"]
        self.dense = SimpleNamespace(d=primary.manifest["dimension"])
        self.languages = Languages(catalog)
        self.doc_ids = catalog.documents
        self.exclusions,self.local,self.eligibility = {},OrderedDict(),{}
        self.manifest = manifest

    def eligible(self,doc_id):
        if doc_id not in self.eligibility:
            doc = self.catalog.documents[doc_id]
            reason = error_page_reason(doc.get("title",""),doc["source_text"])
            if has_encoded_payload(doc["source_text"]):
                reason = "ENCODED_PAYLOAD_REVIEW"
            if doc.get("url") and doc.get("final_url") and _redirected_to_homepage(doc["url"],doc["final_url"]):
                reason = "ARTICLE_REDIRECTED_TO_HOMEPAGE"
            self.eligibility[doc_id] = not reason
            if reason:
                self.exclusions[doc_id] = reason
        return self.eligibility[doc_id]

    def child_text(self,child_id):
        row = self.catalog.children[child_id]
        return build_sparse_text(row,title=self.catalog.documents[row["doc_id"]].get("title",""),heading=row.get("heading",""))

    @property
    def source_language_counts(self):
        if not hasattr(self,"_language_counts"):
            self._language_counts = dict(self.base.con.execute("""SELECT coalesce(json_extract(payload,'$.language'),'unknown'),
                count(*) FROM documents GROUP BY 1"""))
        return self._language_counts

    def document_candidates(self,query,vector,variants,config):
        orders,weights,values = {},{},{}
        def add(name,hits,weight):
            if weight <= 0:
                return
            hits = [(int(d),float(s)) for d,s in hits if self.eligible(int(d))]
            orders[name],weights[name],values[name] = [d for d,_ in hits],weight,dict(hits)
        scores,positions = self.primary_search[query]
        add("dense",self.base.dense_documents(positions,scores,config.doc_candidate_k),config.dense_weight)
        def second(name,text,weight):
            ss,ii = self.secondary_search[digest_json(text)]
            add(name,self.base.dense_documents(ii,ss,config.doc_candidate_k),weight)
        second("dense_qwen",query,config.second_dense_weight)
        if config.document_dense_weight:
            ss,ii = self.document_search[query]
            add("dense_document",[(self.document_ids[int(i)],s) for i,s in zip(ii,ss,strict=True)],config.document_dense_weight)
        expansion = self.expansions[query]
        for i,text in enumerate(expansion["subqueries"]):
            second(f"subquery_{i}",text,config.subquery_weight)
        if expansion["hyde_en"]:
            second("hyde",expansion["hyde_en"],config.hyde_weight)
        for lang,text in (("vi",query),("en",variants.get("query_en")),("zh",variants.get("query_zh"))):
            if text:
                add(lang,self.base.sparse(text,lang,config.sparse_top_k),getattr(config,lang+"_weight"))
        fused = weighted_rrf(orders,weights,config.rrf_k)
        ids = sorted(fused,key=lambda d:(-fused[d],d))[:config.doc_candidate_k]
        ranks = {name:{d:r for r,d in enumerate(order,1)} for name,order in orders.items()}
        rows = [{"doc_id":d,"retrieval_score":fused[d],"branches":{name:{"rank":ranks[name][d],"score":values[name][d]}
            for name in orders if d in ranks[name]}} for d in ids]
        return rows,SimpleNamespace(query=query,vector=np.asarray(vector,np.float32),
            secondary=self.auxiliary[digest_json(query)],global_rank={int(p):i+1 for i,p in enumerate(positions)})

    def child_candidates(self,doc_id,query,variants,unit_scores,config,limit):
        if doc_id not in self.local:
            pairs = self.base.document_children(doc_id)
            rows,positions = zip(*pairs)
            language = self.languages[doc_id]
            texts = [lexical_field(self.base.analyzer,self.child_text(r["chunk_id"]),language) for r in rows]
            self.local[doc_id] = rows,np.asarray(positions,np.int64),PostingBM25(texts)
            if len(self.local) > 64:
                self.local.popitem(last=False)
        self.local.move_to_end(doc_id)
        rows,positions,bm25 = self.local[doc_id]
        language = self.languages[doc_id]
        sparse = bm25.scores(lexical_field(self.base.analyzer,query,language))
        if variants.get("query_"+language):
            sparse = np.maximum(sparse,bm25.scores(lexical_field(self.base.analyzer,variants["query_"+language],language)))
        dense = self.primary.gather(positions) @ unit_scores.vector
        secondary = self.secondary.gather(positions) @ unit_scores.secondary
        order = lambda scores:sorted(range(len(rows)),key=lambda i:(-float(scores[i]),rows[i]["chunk_id"]))
        orders = {"dense":order(dense),"sparse":[i for i in order(sparse) if sparse[i] > 0]}
        weights = {"dense":1.,"sparse":.8}
        if config.second_dense_weight:
            orders["dense_qwen"],weights["dense_qwen"] = order(secondary),config.second_dense_weight
        fused = weighted_rrf(orders,weights,config.rrf_k)
        return [{"child_id":rows[i]["chunk_id"],"doc_id":doc_id,"retrieval_score":fused[i],
            "dense_score":float(dense[i]),"sparse_score":float(sparse[i]),
            "global_dense_rank":unit_scores.global_rank.get(int(positions[i]),self.primary.manifest["input_count"]+1)}
            for i in sorted(fused,key=lambda i:(-fused[i],rows[i]["chunk_id"]))[:limit]]
