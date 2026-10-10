"""Retrieval timing and partial-judgment diagnostics, separate from reranking."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from .index import ProductIndex


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered)-1)*p
    lower = int(index)
    upper = min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower)


def benchmark_index(path: Path, queries: list[dict], k: int = 100) -> dict:
    if not queries:
        raise ValueError("queries must not be empty")
    samples = []
    coverage = []
    observed_recall = []
    rows = []
    with ProductIndex(path,cache_size=0) as index:
        # Warm sqlite pages, then time one pass; query LRU is disabled.
        for row in queries[:5]:
            index.search(row["query"],k)
        for row in queries:
            start = time.perf_counter()
            hits = index.search(row["query"],k)
            samples.append(1000*(time.perf_counter()-start))
            ids = {p["id"] for p in hits}
            gold = row["qrels"]
            positives = {pid for pid,grade in gold.items() if grade>=2}
            recall = len(ids & positives)/len(positives) if positives else None
            judged = len(ids & set(gold))/len(ids) if ids else None
            if recall is not None: observed_recall.append(recall)
            if judged is not None: coverage.append(judged)
            rows.append({"query_id":row["query_id"],"returned":len(ids),f"observed_positive_recall_at_{k}":recall,"judged_coverage":judged,"latency_ms":samples[-1]})
        return {"backend":"SQLite FTS5; title+brand; shared lexical rules; no hard word-NOT","scope":f"{index.metadata['products']}-product union of selected ESCI judged products; per-query judgments incomplete, not full Amazon catalog","products":int(index.metadata["products"]),"catalog_sha256":index.metadata["catalog_sha256"],"queries":len(queries),"k":k,"index_bytes":Path(path).stat().st_size,"sqlite_page_cache_limit_bytes":8*1024*1024,"query_result_cache_enabled":False,"latency_protocol":"5-query page warmup; one sequential pass; perf_counter around retrieval only; includes product text fetch; excludes model load and reranking","p50_ms":percentile(samples,.5),"p95_ms":percentile(samples,.95),f"observed_positive_recall_at_{k}":statistics.mean(observed_recall) if observed_recall else None,"mean_judged_coverage":statistics.mean(coverage) if coverage else None,"per_query":rows}
