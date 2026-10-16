"""Small sealed cursors; reused payload SHA checks happen in native consumers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import pyarrow.parquet as pq

from .artifacts import atomic_json, digest_json, read_json, verify_file

FIELDS = ("signature","start","rows","path","sha256")
FORMAT = "index-input-cursor-v1"


def valid_key(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def metadata_parts(parts, signature, part_size, names, *, total=None):
    cursor, result = 0,[]
    for item in parts:
        part = {k:item[k] for k in FIELDS}
        expected = part_size if total is None else min(part_size,total-cursor)
        if (part["signature"] != signature or type(part["start"]) is not int or part["start"] != cursor
            or type(part["rows"]) is not int or not 0 < part["rows"] <= expected
            or (total is not None and part["rows"] != expected)
            or part["path"] != f"units-{cursor:012d}.parquet" or not valid_key(part["sha256"])):
            raise ValueError("Index input resume metadata changed.")
        if (part["path"] not in names or f"units-{cursor:012d}.done.json" not in names):
            raise ValueError("Missing index input checkpoint payload/receipt; restore the artifact before resume.")
        result.append(part)
        cursor += part["rows"]
    return result,cursor


class InputResume:
    def __init__(self, target, signature, part_size, *, expected_total=None):
        self.target,self.signature,self.part_size = Path(target),signature,part_size
        self.expected_total = expected_total
        self.path = self.target / "resume.json"
        # One directory listing avoids thousands of independent Drive stat calls.
        self.names = {p.name for p in self.target.iterdir()}
        self.parts,self.cursor,self.last = [],0,""
        self.spool_base_id,self.spool_base_rows = None,0

    def load(self):
        if self.path.name in self.names:
            saved = read_json(self.path)
            if (saved.get("format") != FORMAT or saved.get("signature") != self.signature
                or digest_json({k:v for k,v in saved.items() if k != "manifest_sha256"}) != saved.get("manifest_sha256")
                or (self.expected_total is not None and saved.get("total") != self.expected_total)):
                raise ValueError("Index input resume cursor changed.")
            self.parts,self.cursor = metadata_parts(saved["parts"],self.signature,self.part_size,self.names,
                                                    total=saved["total"])
            self.last = saved["last_id"]
            self.spool_base_id,self.spool_base_rows = saved["spool_base_id"],saved["spool_base_rows"]
            if (saved["cursor"] != self.cursor or self.cursor > saved["total"]
                or (not valid_key(self.last) if self.cursor else self.last != "")
                or type(self.spool_base_rows) is not int or not 0 <= self.spool_base_rows <= self.cursor
                or (self.spool_base_id is not None and (not valid_key(self.spool_base_id) or self.spool_base_id > self.last))
                or (self.spool_base_id is None and self.spool_base_rows != 0)):
                raise ValueError("Index input resume cursor boundary changed.")
        else:
            markers = sorted(n for n in self.names if re.fullmatch(r"units-[0-9]{12}\.done\.json",n))
            receipts = {}
            if markers:
                print(f"Import legacy input checkpoints: {len(markers):,} small receipts; no old text payload scan",flush=True)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = {pool.submit(read_json,self.target / n):n for n in markers}
                    for count,future in enumerate(as_completed(futures),1):
                        name = futures[future]
                        item = future.result()
                        if item.get("start") != int(name[6:18]) or item.get("signature") != self.signature:
                            raise ValueError("Legacy input checkpoint source changed.")
                        receipts[item["start"]] = item
                        if count % 128 == 0 or count == len(markers):
                            print(f"Input checkpoint metadata: {count:,}/{len(markers):,}",flush=True)
                prefix = []
                while self.cursor in receipts:
                    item = receipts[self.cursor]
                    if type(item.get("rows")) is not int or item["rows"] < 1:
                        raise ValueError("Invalid legacy input checkpoint rows.")
                    prefix.append(item);self.cursor += item["rows"]
                self.parts,self.cursor = metadata_parts(prefix,self.signature,self.part_size,self.names,
                                                        total=self.expected_total)
                if self.parts:
                    # Old receipts lack a last-key field. Read just the boundary
                    # part once to migrate them, not every saved text file.
                    boundary = self.parts[-1]
                    path = self.target / boundary["path"]
                    verify_file(path,boundary["sha256"])
                    ids = pq.read_table(path,columns=["id"]).column("id").to_pylist()
                    if (len(ids) != boundary["rows"] or any(not valid_key(k) for k in ids)
                        or ids != sorted(set(ids))):
                        raise ValueError("Legacy input boundary changed.")
                    self.last = ids[-1]
        if self.cursor:
            print(f"Input resume cursor: {self.cursor:,} texts, {len(self.parts):,} completed parts"
                  " | old payload verification deferred until consumption",flush=True)
        return self

    def save(self, parts, last, total, *, spool_base_id=None, spool_base_rows=0):
        saved = {"format":FORMAT,"signature":self.signature,"parts":parts,
            "cursor":sum(p["rows"] for p in parts),"last_id":last,"total":total,
            "spool_base_id":spool_base_id,"spool_base_rows":spool_base_rows,
            "payload_verification":"VERIFY_SHA256_ON_USE"}
        saved["manifest_sha256"] = digest_json(saved)
        atomic_json(self.path,saved)
