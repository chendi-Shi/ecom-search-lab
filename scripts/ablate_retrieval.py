"""Retrospective retrieval component study; no training or test-driven selection.

prepare freezes the protocol before evaluate scores the already-seen public test.
The five frozen contrasts are diagnostic, not fresh generalization evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecom_search.data import file_sha256, load_jsonl, save_json, validate_dataset
from ecom_search.index import ProductIndex, build_index, catalog_digest
from ecom_search.query import QueryPlan, plan_query
from ecom_search.ranking import FeatureExtractor, PairwiseRanker, _tokens
from scripts.benchmark_public import (_assert_fingerprints, _corpus_metrics, _fingerprints,
                                     _latency, _mean, _rank, _ranking_query, paired_bootstrap)

VARIANTS = {
    "full_k100": {"pool": 100, "planner": "full", "index": "original"},
    "no_brand_index_k100": {"pool": 100, "planner": "full", "index": "title_only"},
    "no_exclusion_filter_k100": {"pool": 100, "planner": "no_exclusion", "index": "original"},
    "raw_tokens_k100": {"pool": 100, "planner": "raw", "index": "original"},
    "full_k50": {"pool": 50, "planner": "full", "index": "original"},
    "full_k200": {"pool": 200, "planner": "full", "index": "original"},
}
SOURCE_FILES = ("scripts/ablate_retrieval.py", "scripts/benchmark_public.py",
                "ecom_search/data.py", "ecom_search/index.py", "ecom_search/query.py",
                "ecom_search/ranking.py", "ecom_search/evaluate.py")
METRICS = ("known_positive_recall@10", "known_positive_recall@100",
           "candidate_known_positive_recall", "judged_fraction@10")


def now():
    return datetime.now(timezone.utc).isoformat()


def make_plan(query, mode):
    if mode == "raw":
        # Same lexical tokenizer as ranking. This removes stopword/negation and
        # leading-minus handling as a bundle, not just one parsing subfeature.
        return QueryPlan(query, tuple(dict.fromkeys(_tokens(query))), (), (), (), (), _ranking_query(query))
    try:
        plan = plan_query(query)
    except ValueError:
        return None
    return replace(plan, excluded_terms=()) if mode == "no_exclusion" else plan


def retrieve(index, plan, k, original_catalog):
    """Exact FTS primitive with original metadata restored before constraints.

    For no_brand_index the separate index has blank brands. This removes brand
    TF/DF/length/match signals; original brand values remain available to the
    unchanged exclusion rule and reranker, isolating the retrieval field.
    """
    if plan is None or not plan.positive_terms:
        return []
    quote = lambda token: '"' + token.replace('"', '""') + '"'
    match = "(" + " OR ".join(map(quote, plan.positive_terms)) + ")"
    depth = min(1000, k * 3) if plan.excluded_terms else k
    rows = index.connection.execute(
        "SELECT p.id,-bm25(terms,3.0,1.0) AS retrieval_score FROM terms "
        "JOIN products p ON p.rowid=terms.rowid WHERE terms MATCH ? "
        "ORDER BY bm25(terms,3.0,1.0),p.id LIMIT ?", (match, depth)).fetchall()
    hits = [dict(original_catalog[row["id"]], retrieval_score=row["retrieval_score"]) for row in rows]
    if plan.excluded_terms:
        forbidden = set(plan.excluded_terms)
        hits = [p for p in hits if not forbidden & FeatureExtractor.affirmed_product_terms(p)]
    return hits[:k]


def metrics(ids, qrels):
    result = _corpus_metrics(ids, qrels)
    positives = {pid for pid, grade in qrels.items() if grade >= 2}
    result["candidate_known_positive_recall"] = len(set(ids) & positives) / len(positives) if positives else None
    return result


def prepare(data, reference, output):
    data, reference, output = map(Path, (data, reference, output))
    if output.exists():
        raise FileExistsError("Choose a new output directory; no prior run is overwritten")
    old_freeze = json.loads((reference / "freeze.json").read_text(encoding="utf-8"))
    if old_freeze.get("protocol") != "public_cpu_v1":
        raise ValueError("Reference must be the frozen public CPU experiment")
    _assert_fingerprints(data, old_freeze["data_fingerprints"], "reference data")
    _assert_fingerprints(reference, old_freeze["artifact_fingerprints"], "reference artifact")
    _assert_fingerprints(ROOT, old_freeze["source_fingerprints"], "reference source")
    catalog = load_jsonl(data / "catalog.jsonl")
    model = json.loads((reference / "model.json").read_text(encoding="utf-8"))
    if model["catalog_sha256"] != catalog_digest(catalog):
        raise ValueError("Model/catalog mismatch")
    reference_names = ["model.json", "products.sqlite", "freeze.json", "report.json"]
    if (reference / "per_query_test.jsonl").is_file():
        reference_names.append("per_query_test.jsonl")
    source_hashes = _fingerprints(ROOT, SOURCE_FILES)
    output.mkdir(parents=True)
    build_index([dict(p, brand="") for p in catalog], output / "title_only.sqlite")
    _assert_fingerprints(ROOT, source_hashes, "source during preparation")
    protocol = {
        "protocol": "retrieval_ablation_v1", "created_at_utc": now(),
        "study_type": "retrospective diagnostic on previously reported test; no hyperparameter selection",
        "variants": VARIANTS, "reference": "full_k100", "difference_direction": "full_k100 minus ablated",
        "data_fingerprints": old_freeze["data_fingerprints"],
        "reference_fingerprints": _fingerprints(reference, reference_names),
        "source_fingerprints": source_hashes,
        "artifact_fingerprints": _fingerprints(output, ("title_only.sqlite",)),
        "dataset": old_freeze["dataset"], "synthetic": old_freeze["synthetic"],
        "parameters": {"model": "unchanged validation-selected reference pairwise", "bootstrap_samples": 2000,
                       "bootstrap_seed": 20261010, "application_result_cache": False,
                       "query_feature_cache": "cleared before each timed variant", "method_order": "cyclic rotation by query index"},
        "test_labels_parsed_in_preparation": False,
        "limits": ["Different retrieval pools; separate from identical-pool ranker ablations",
                   "Removing brand rebuilds field TF/DF/document-length statistics, not only a BM25 weight",
                   "Disabling exclusion removes its 3x oversampling-and-filtering module",
                   "Raw-token mode bundles stopword/negation/minus processing; tokenizer unchanged",
                   "Recall@100 for pool50 is capped at 50 returned items; candidate recall uses entire pool",
                   "Intervals are exploratory pointwise query-bootstrap; no multiple-comparison correction or training-seed variance",
                   "Known-positive recall on finite judged-product union; unjudged pairs unknown"],
        "planned_subset": "Queries with exclusions under the full parser; full vs no-exclusion only, exploratory",
    }
    save_json(output / "protocol.json", protocol)
    return protocol


def evaluate(data, reference, output):
    data, reference, output = map(Path, (data, reference, output))
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol") != "retrieval_ablation_v1" or protocol.get("variants") != VARIANTS:
        raise ValueError("Unexpected or changed protocol")
    for directory, key in ((data, "data_fingerprints"), (reference, "reference_fingerprints"),
                           (ROOT, "source_fingerprints"), (output, "artifact_fingerprints")):
        _assert_fingerprints(directory, protocol[key], key)
    with (output / "evaluation_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": now(), "protocol_sha256": file_sha256(output / "protocol.json")}, handle)
    started = time.perf_counter()
    catalog, records = load_jsonl(data / "catalog.jsonl"), load_jsonl(data / "test.jsonl")
    validate_dataset(catalog, {"test": records})
    extractor = FeatureExtractor(catalog)
    model = PairwiseRanker.from_dict(json.loads((reference / "model.json").read_text(encoding="utf-8"))["ranker"])
    original_rows = {}
    if "per_query_test.jsonl" in protocol["reference_fingerprints"]:
        original_rows = {r["query_id"]: r for r in load_jsonl(reference / "per_query_test.jsonl")}
    if original_rows and set(original_rows) != {r["query_id"] for r in records}:
        raise ValueError("Reference per-query identities differ")
    names = list(VARIANTS)
    rows, baseline_checks = [], 0
    with ProductIndex(reference / "products.sqlite", cache_size=0) as full_index, ProductIndex(output / "title_only.sqlite", cache_size=0) as title_index:
        for number, row in enumerate(records):
            query, qrels = row["query"], row.get("full_qrels", row["qrels"])
            item = {"query_id": row["query_id"], "has_exclusion": bool(getattr(make_plan(query, "full"), "excluded_terms", ())), "methods": {}}
            order = names[number % len(names):] + names[:number % len(names)]
            for name in order:
                variant = VARIANTS[name]
                plan = make_plan(query, variant["planner"])
                extractor._query_stats.cache_clear()
                tick = time.perf_counter()
                index = title_index if variant["index"] == "title_only" else full_index
                hits = retrieve(index, plan, variant["pool"], extractor.products)
                retrieval_ms = 1000 * (time.perf_counter() - tick)
                tick = time.perf_counter()
                ids = [p["id"] for p in hits]
                ranked = _rank(ids, [model.score(extractor.features(_ranking_query(query), p)) for p in hits])
                rerank_ms = 1000 * (time.perf_counter() - tick)
                if name == "full_k100":
                    expected = full_index.search(plan, 100) if plan is not None else []
                    if ids != [p["id"] for p in expected] or any(abs(a["retrieval_score"]-b["retrieval_score"]) > 1e-12 for a, b in zip(hits, expected)):
                        raise ValueError("Full primitive differs from ProductIndex")
                    if original_rows:
                        historical = original_rows[row["query_id"]]["rankings"]
                        if ids != historical["retrieval"] or ranked != historical["reranked"]:
                            raise ValueError("Frozen reference rankings failed exact reproduction")
                    baseline_checks += 1
                item["methods"][name] = {"retrieval_ids": ids, "reranked_ids": ranked,
                    "retrieval": metrics(ids, qrels), "reranked": metrics(ranked, qrels),
                    "candidates": len(ids), "retrieval_ms": retrieval_ms, "rerank_ms": rerank_ms}
            rows.append(item)
            if (number+1) % 25 == 0:
                print(f"retrieval ablation evaluated {number+1}/{len(records)} queries", flush=True)
    summary, comparisons = {}, {}
    for name in names:
        method_rows = [r["methods"][name] for r in rows]
        summary[name] = {scope: {m: _mean(r[scope][m] for r in method_rows) for m in METRICS} for scope in ("retrieval", "reranked")}
        summary[name].update(candidate_count=_mean(r["candidates"] for r in method_rows),
                             changed_retrieval_order_queries=sum(r["methods"][name]["retrieval_ids"] != r["methods"]["full_k100"]["retrieval_ids"] for r in rows),
                             latency={"retrieval": _latency([r["retrieval_ms"] for r in method_rows]),
                                      "retrieval_and_rerank": _latency([r["retrieval_ms"]+r["rerank_ms"] for r in method_rows])})
        if name != "full_k100":
            comparisons[name] = {scope: {m: paired_bootstrap(
                [r["methods"][name][scope][m] for r in rows], [r["methods"]["full_k100"][scope][m] for r in rows])
                for m in METRICS} for scope in ("retrieval", "reranked")}
    historical = json.loads((reference / "report.json").read_text(encoding="utf-8"))
    affected = [r for r in rows if r["has_exclusion"]]
    exclusion_subset = {"queries": len(affected), "full_minus_no_exclusion": {
        scope: {m: paired_bootstrap(
            [r["methods"]["no_exclusion_filter_k100"][scope][m] for r in affected],
            [r["methods"]["full_k100"][scope][m] for r in affected]) for m in METRICS}
        for scope in ("retrieval", "reranked")}}
    for scope, method in (("retrieval", "fts_bm25"), ("reranked", "fts_pairwise")):
        for metric in ("known_positive_recall@10", "known_positive_recall@100", "judged_fraction@10"):
            actual, expected = summary["full_k100"][scope][metric]["mean"], historical["corpus"][method][metric]["mean"]
            if actual is None or expected is None:
                if actual != expected: raise ValueError("Reference denominator reproduction failed")
            elif abs(actual-expected) > 1e-12:
                raise ValueError("Reference metric reproduction failed")
    for directory, key in ((data, "data_fingerprints"), (reference, "reference_fingerprints"),
                           (ROOT, "source_fingerprints"), (output, "artifact_fingerprints")):
        _assert_fingerprints(directory, protocol[key], key)
    report = {"protocol": protocol["protocol"], "study_type": protocol["study_type"], "created_at_utc": now(),
              "protocol_sha256": file_sha256(output / "protocol.json"), "catalog_products": len(catalog),
              "test_queries": len(records), "source_fingerprints": protocol["source_fingerprints"],
              "data_fingerprints": protocol["data_fingerprints"], "model_sha256": protocol["reference_fingerprints"]["model.json"],
              "baseline_reproduction": {"queries": baseline_checks, "exact_historical_rankings_checked": bool(original_rows), "metrics_tolerance": 1e-12},
              "activity": {"queries_with_exclusion": sum(r["has_exclusion"] for r in rows), "nonempty_description_products": sum(bool(p.get("description")) for p in catalog)},
              "variants": VARIANTS, "methods": summary, "full_minus_variant": comparisons,
              "exclusion_subset": exclusion_subset,
              "elapsed_seconds": time.perf_counter()-started, "latency_scope": "descriptive timings on a loaded machine; retrieval measures FTS SQL plus constraint filtering, excludes query planning; model/catalog resident, application result cache disabled, query feature cache cleared per variant; OS caches uncontrolled; cyclic method order; no speedup claim",
              "limits": protocol["limits"], "default_model_changed": False}
    save_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "evaluate"))
    p.add_argument("--data", type=Path, default=Path("data/esci-public"))
    p.add_argument("--reference", type=Path, default=Path("outputs/public-cpu-v1"))
    p.add_argument("--output", type=Path, default=Path("outputs/ablation-retrieval-v1"))
    args = p.parse_args()
    result = (prepare if args.phase == "prepare" else evaluate)(args.data, args.reference, args.output)
    print(json.dumps({"phase": args.phase, "protocol": result["protocol"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
