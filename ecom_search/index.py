"""SQLite FTS5 colocates catalog and inverted index in one persistent file.

This is a CPU lexical index, not an ANN or billion-item deployment claim.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .query import QueryPlan, plan_query

INDEX_VERSION = 1


def catalog_digest(catalog: list[dict]) -> str:
    canonical = [{k: row.get(k, "") for k in ("id", "title", "brand", "description")} for row in catalog]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_index(catalog: list[dict], path: str | Path) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Index exists: {path}; choose a new output path")
    ids = [p["id"] for p in catalog]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate product IDs")
    if not catalog:
        raise ValueError("catalog is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE products(id TEXT UNIQUE NOT NULL,title TEXT NOT NULL,brand TEXT NOT NULL,description TEXT NOT NULL)")
        conn.execute("CREATE VIRTUAL TABLE terms USING fts5(title,brand,content='products',content_rowid='rowid',tokenize='unicode61 remove_diacritics 2')")
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        conn.executemany("INSERT INTO products VALUES(?,?,?,?)", [(p["id"],p["title"],p.get("brand", ""),p.get("description", "")) for p in catalog])
        conn.execute("INSERT INTO terms(terms) VALUES('rebuild')")
        conn.execute("INSERT INTO terms(terms) VALUES('optimize')")
        digest = catalog_digest(catalog)
        conn.executemany("INSERT INTO metadata VALUES(?,?)", [("catalog_sha256",digest),("index_version",str(INDEX_VERSION)),("products",str(len(catalog)))])
        conn.commit()
    finally:
        conn.close()
    return {"path":str(path),"products":len(catalog),"catalog_sha256":digest,"index_bytes":path.stat().st_size,"build_seconds":time.perf_counter()-start,"backend":"SQLite FTS5 title+brand BM25"}


class ProductIndex:
    def __init__(self, path: str | Path, *, cache_size: int = 128):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Build the index first: {path}")
        self.connection = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA cache_size=-8192")
        self.metadata = dict(self.connection.execute("SELECT key,value FROM metadata"))
        if self.metadata.get("index_version") != str(INDEX_VERSION):
            self.connection.close()
            raise ValueError("unsupported index version")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.cache_size = cache_size
        self.cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self.cache_hits = 0

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def catalog(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT id,title,brand,description FROM products ORDER BY rowid")]

    def get_products(self, ids: list[str]) -> dict[str,dict]:
        if not ids:
            return {}
        if len(ids)>1000:
            raise ValueError("at most 1000 products per lookup")
        placeholders = ",".join("?" for _ in ids)
        return {row["id"]:dict(row) for row in self.connection.execute(f"SELECT id,title,brand,description FROM products WHERE id IN ({placeholders})",ids)}

    def search(self, query: str | QueryPlan, k: int = 100) -> list[dict]:
        if not isinstance(k,int) or isinstance(k,bool) or not 1 <= k <= 1000:
            raise ValueError("k must be an integer in [1, 1000]")
        plan = plan_query(query) if isinstance(query,str) else query
        if not plan.positive_terms:
            return []
        # Quote each token: user text cannot become FTS syntax or SQL code.
        quote = lambda t: '"'+t.replace('"','""')+'"'
        match = "(" + " OR ".join(map(quote, plan.positive_terms)) + ")"
        # Do not hard-NOT forbidden words: that loses e.g. 'sugar free snack'.
        # Filter affirmed catalog terms below; keep explicitly negated terms.
        key = (match,plan.excluded_terms,k)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return [dict(p) for p in self.cache[key]]
        retrieval_k = min(1000,k*3) if plan.excluded_terms else k
        rows = self.connection.execute("SELECT p.id,p.title,p.brand,p.description,-bm25(terms,3.0,1.0) AS retrieval_score FROM terms JOIN products p ON p.rowid=terms.rowid WHERE terms MATCH ? ORDER BY bm25(terms,3.0,1.0),p.id LIMIT ?",(match,retrieval_k)).fetchall()
        result = [dict(row) for row in rows]
        if plan.excluded_terms:
            from .ranking import FeatureExtractor
            forbidden = set(plan.excluded_terms)
            result = [p for p in result if not forbidden & FeatureExtractor.affirmed_product_terms(p)]
        result = result[:k]
        if self.cache_size:
            self.cache[key] = result
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return [dict(p) for p in result]
