"""Independent CPU benchmark: validation selection, then an explicit sealed test.

Only Python's standard library is required. The public-data importer uses
pyarrow separately. Historical project artifacts are neither loaded nor needed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_search.data import (file_sha256, load_jsonl, normalized_query, save_json,
                             save_jsonl, validate_dataset)
from ecom_search.evaluate import ndcg, recall
from ecom_search.index import ProductIndex, build_index, catalog_digest
from ecom_search.query import plan_query
from ecom_search.ranking import FeatureExtractor, PairwiseRanker

EPOCH_GRID = (0, 5, 15, 30)
DATA_FILES = ("catalog.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "manifest.json")
SOURCE_FILES = ("scripts/benchmark_public.py", "scripts/prepare_public_esci.py", "ecom_search/data.py",
                "ecom_search/ranking.py", "ecom_search/evaluate.py",
                "ecom_search/query.py", "ecom_search/index.py", "ecom_search/cli.py")
FIXED_METRICS = ("ndcg@10", "p@10_es", "exact@1", "known_positive_recall@10")
CORPUS_METRICS = ("known_positive_recall@10", "known_positive_recall@100",
                  "judged_fraction@10", "any_known_positive@100")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprints(directory, names):
    return {name: file_sha256(Path(directory) / name) for name in names}


def _assert_fingerprints(directory, expected, label):
    for name, digest in expected.items():
        target = Path(directory) / name
        if not target.is_file() or file_sha256(target) != digest:
            raise ValueError(f"{label} fingerprint changed: {name}")


def _manifest(directory):
    value = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    for name in DATA_FILES[:-1]:
        entry = value.get("files", {}).get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"manifest must fingerprint {name}")
        target = Path(directory) / name
        if file_sha256(target) != entry["sha256"] or target.stat().st_size != entry.get("bytes"):
            raise ValueError(f"manifest mismatch: {name}")
    return value


def _ranking_query(query):
    # Identical query normalization to SearchEngine for supported queries.
    # A stopword-only query has no retrieval hits but remains in the evaluation.
    try:
        return plan_query(query).ranking_query
    except ValueError:
        return normalized_query(query)


def _rank(ids, scores):
    if len(ids) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError("one finite score is required for each candidate")
    return [pid for pid, _ in sorted(zip(ids, scores), key=lambda p: (-p[1], p[0]))]


def _mean(values):
    values = [value for value in values if value is not None]
    return {"mean": statistics.fmean(values) if values else None, "queries": len(values)}


def _summary(rows, metrics):
    return {name: _mean(row[name] for row in rows) for name in metrics}


def _fixed_metrics(ranking, qrels):
    return {
        "ndcg@10": ndcg(ranking, qrels, 10),
        # The importer requires >=20 judgments/query; shorter custom fixtures
        # still use the precision@10 denominator rather than precision@min(n,10).
        "p@10_es": sum(qrels[pid] >= 2 for pid in ranking[:10]) / 10,
        "exact@1": float(bool(ranking) and qrels[ranking[0]] == 3),
        "known_positive_recall@10": recall(ranking, qrels, 10),
    }


def _corpus_metrics(ranking, qrels):
    positives = {pid for pid, grade in qrels.items() if grade >= 2}
    return {
        "known_positive_recall@10": recall(ranking, qrels, 10),
        "known_positive_recall@100": recall(ranking, qrels, 100),
        # Coverage of an incomplete judgment set; this is not relevance precision.
        "judged_fraction@10": sum(pid in qrels for pid in ranking[:10]) / 10,
        "any_known_positive@100": float(bool(set(ranking[:100]) & positives)) if positives else None,
    }


def paired_bootstrap(before, after, *, samples=2000, seed=20261010):
    """Paired query-level percentile bootstrap; do not infer online impact."""
    if len(before) != len(after) or samples < 100:
        raise ValueError("paired arrays must have equal length and samples must be >=100")
    deltas = [b - a for a, b in zip(before, after) if a is not None and b is not None]
    if not deltas:
        return {"queries": 0, "delta": None, "ci95": None, "samples": samples, "seed": seed}
    rng = random.Random(seed)
    estimates = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    lower, upper = estimates[int((samples - 1) * .025)], estimates[int((samples - 1) * .975)]
    return {"queries": len(deltas), "delta": statistics.fmean(deltas), "ci95": [lower, upper],
            "samples": samples, "seed": seed, "interval_excludes_zero": lower > 0 or upper < 0,
            "method": "paired query bootstrap percentile; exploratory; no multiple-comparison correction"}


def _paired(rows, field, before, after, metrics):
    return {name: paired_bootstrap([r[field][before][name] for r in rows],
                                  [r[field][after][name] for r in rows]) for name in metrics}


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[math.ceil(fraction * len(values)) - 1]


def _latency(values):
    return {"queries": len(values), "p50_ms": _percentile(values, .50),
            "p95_ms": _percentile(values, .95), "mean_ms": statistics.fmean(values) if values else None}


def select(data, output):
    """Parse train/val only. Test is hashed as opaque bytes, never JSON-decoded."""
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError("Selection output already exists; use a new run directory")
    started = time.perf_counter()
    manifest = _manifest(data)
    fingerprints = _fingerprints(data, DATA_FILES)
    source_fingerprints = _fingerprints(ROOT, SOURCE_FILES)
    catalog = load_jsonl(data / "catalog.jsonl")
    train, val = load_jsonl(data / "train.jsonl"), load_jsonl(data / "val.jsonl")
    validate_dataset(catalog, {"train": train, "val": val})
    if not train or not val:
        raise ValueError("train and validation must both be nonempty")
    # Keep train/val query identities to check holdout overlap before final scoring.
    identities = {"query_ids": sorted({r["query_id"] for r in train + val}),
                  "normalized_queries": sorted({normalized_query(r["query"]) for r in train + val})}
    extractor = FeatureExtractor(catalog)
    validation_features = [
        [extractor.features(_ranking_query(r["query"]), extractor.products[pid])
         for pid in r["candidate_ids"]] for r in val]
    normalized_train = [dict(row, query=_ranking_query(row["query"])) for row in train]
    baseline = []
    for row in val:
        scores = [extractor.bm25(_ranking_query(row["query"]), extractor.products[pid])
                  for pid in row["candidate_ids"]]
        baseline.append(_fixed_metrics(_rank(row["candidate_ids"], scores), row["qrels"]))
    experiments, models = [], {}
    for epochs in EPOCH_GRID:
        candidate = PairwiseRanker()
        fit_start = time.perf_counter()
        training = candidate.fit(normalized_train, extractor, epochs=epochs, seed=42,
                                 learning_rate=.05, l2=.001)
        if training["pairs"] == 0:
            raise ValueError("train data has no unequal-grade judged pairs")
        metrics = [_fixed_metrics(_rank(row["candidate_ids"], [candidate.score(f) for f in features]), row["qrels"])
                   for row, features in zip(val, validation_features)]
        experiments.append({"epochs": epochs, "training": training,
                            "fit_and_validation_seconds": time.perf_counter() - fit_start,
                            "validation": _summary(metrics, FIXED_METRICS)})
        models[epochs] = candidate
        print(f"validated epochs={epochs} ndcg@10={experiments[-1]['validation']['ndcg@10']['mean']}",
              file=sys.stderr, flush=True)
    eligible = [r for r in experiments if r["validation"]["ndcg@10"]["mean"] is not None]
    if not eligible:
        raise ValueError("validation contains no query with positive ideal DCG")
    chosen = max(eligible, key=lambda r: (r["validation"]["ndcg@10"]["mean"], -r["epochs"]))
    _assert_fingerprints(data, fingerprints, "data")
    _assert_fingerprints(ROOT, source_fingerprints, "source")
    output.mkdir(parents=True, exist_ok=False)
    model = {"ranker": models[chosen["epochs"]].to_dict(), "catalog_sha256": catalog_digest(catalog),
             "selected_epochs": chosen["epochs"], "frozen_before_test": True,
             "data_fingerprints": fingerprints, "protocol": "public_cpu_v1"}
    save_json(output / "model.json", model)
    index = build_index(catalog, output / "products.sqlite")
    index.pop("path", None)
    selection = {"schema_version": 1, "created_at_utc": _now(), "catalog_products": len(catalog),
                 "split_queries": {"train": len(train), "val": len(val)},
                 "test_labels_decoded": False, "selected_epochs": chosen["epochs"],
                 "criterion": "maximum validation macro NDCG@10; tie chooses fewer epochs",
                 "grid": list(EPOCH_GRID), "candidates": experiments,
                 "title_bm25_validation": _summary(baseline, FIXED_METRICS),
                 "index": index, "selection_seconds": time.perf_counter() - started}
    save_json(output / "selection.json", selection)
    save_json(output / "train_val_identities.json", identities)
    artifact_files = ("model.json", "products.sqlite", "selection.json", "train_val_identities.json")
    freeze = {"schema_version": 1, "protocol": "public_cpu_v1", "frozen_at_utc": _now(),
              "data_fingerprints": fingerprints, "source_fingerprints": source_fingerprints,
              "artifact_fingerprints": _fingerprints(output, artifact_files),
              "test_labels_decoded": False, "retrieval_pool": 100, "bootstrap_samples": 2000,
              "dataset": manifest.get("dataset", "user_supplied"),
              "synthetic": manifest.get("synthetic"),
              "test_policy": "one final command per frozen directory; no test-dependent model selection"}
    save_json(output / "freeze.json", freeze)
    return selection


def _slice_tags(query, brands):
    words = re.findall(r"[^\W_]+(?:\.[0-9]+)?", normalized_query(query))
    padded = " " + " ".join(words) + " "
    tags = ["long_ge5_tokens" if len(words) >= 5 else "short_lt5_tokens",
            "contains_number" if any(c.isdigit() for c in query) else "no_number"]
    tags.append("literal_catalog_brand" if any(f" {brand} " in padded for brand in brands)
                else "no_literal_catalog_brand")
    return tags


def final(data, output, *, per_query=False):
    """Reject any post-freeze change before loading test labels or scoring."""
    data, output = Path(data), Path(output)
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    if freeze.get("protocol") != "public_cpu_v1":
        raise ValueError("unsupported frozen protocol")
    _assert_fingerprints(data, freeze["data_fingerprints"], "data")
    _assert_fingerprints(ROOT, freeze["source_fingerprints"], "source")
    _assert_fingerprints(output, freeze["artifact_fingerprints"], "artifact")
    if (output / "report.json").exists():
        raise FileExistsError("Final report already exists; the holdout was already consumed")
    # Exclusive marker prevents rerunning final after a crash or a bad outcome.
    # This is an auditable local guard, not access control against malicious edits.
    with (output / "final_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": _now(), "freeze_sha256": file_sha256(output / "freeze.json")}, handle)
    started = time.perf_counter()
    catalog = load_jsonl(data / "catalog.jsonl")
    test = load_jsonl(data / "test.jsonl")
    validate_dataset(catalog, {"test": test})
    if not test:
        raise ValueError("test must be nonempty")
    identities = json.loads((output / "train_val_identities.json").read_text(encoding="utf-8"))
    old_ids, old_queries = set(identities["query_ids"]), set(identities["normalized_queries"])
    if any(row["query_id"] in old_ids or normalized_query(row["query"]) in old_queries for row in test):
        raise ValueError("holdout overlaps train/validation by ID or normalized query")
    for row in test:
        full = row.get("full_qrels", row["qrels"])
        if any(full.get(pid) != grade for pid, grade in row["qrels"].items()):
            raise ValueError("full_qrels must include all candidate judgments unchanged")
    obj = json.loads((output / "model.json").read_text(encoding="utf-8"))
    if obj["catalog_sha256"] != catalog_digest(catalog):
        raise ValueError("model and catalog differ")
    model = PairwiseRanker.from_dict(obj["ranker"])
    load_started = time.perf_counter()
    extractor = FeatureExtractor(catalog)
    feature_load_seconds = time.perf_counter() - load_started
    brands = {" ".join(re.findall(r"[^\W_]+(?:\.[0-9]+)?", normalized_query(p.get("brand", ""))))
              for p in catalog if p.get("brand")}
    brands.discard("")
    rows = []
    with ProductIndex(output / "products.sqlite", cache_size=0) as index:
        for number, row in enumerate(test, 1):
            query = _ranking_query(row["query"])
            fixed_ids = row["candidate_ids"]
            retrieval_start = time.perf_counter()
            retrieval_error = None
            try:
                hits = index.search(row["query"], k=freeze["retrieval_pool"])
            except ValueError as exc:
                hits, retrieval_error = [], str(exc)
            retrieved_ids = [hit["id"] for hit in hits]
            retrieval_ms = 1000 * (time.perf_counter() - retrieval_start)
            rerank_start = time.perf_counter()
            reranked_ids = _rank(retrieved_ids, [model.score(extractor.features(query, hit)) for hit in hits])
            rerank_ms = 1000 * (time.perf_counter() - rerank_start)
            # Measure corpus serving before fixed-pool scoring can warm the
            # feature extractor's per-query cache for this query.
            fixed_bm25 = _rank(fixed_ids, [extractor.bm25(query, extractor.products[pid]) for pid in fixed_ids])
            fixed_learned = _rank(fixed_ids, [model.score(extractor.features(query, extractor.products[pid])) for pid in fixed_ids])
            full_qrels = row.get("full_qrels", row["qrels"])
            rows.append({"query_id": row["query_id"], "query": row["query"],
                         "slices": _slice_tags(row["query"], brands), "candidate_count": len(fixed_ids),
                         "retrieved_count": len(retrieved_ids), "retrieval_error": retrieval_error,
                         "fixed_pool": {"title_bm25": _fixed_metrics(fixed_bm25, row["qrels"]),
                                        "learned_pairwise": _fixed_metrics(fixed_learned, row["qrels"])},
                         "corpus": {"fts_bm25": _corpus_metrics(retrieved_ids, full_qrels),
                                    "fts_pairwise": _corpus_metrics(reranked_ids, full_qrels)},
                         "rankings": {"fixed_bm25": fixed_bm25, "fixed_pairwise": fixed_learned,
                                      "retrieval": retrieved_ids, "reranked": reranked_ids},
                         "latency_ms": {"retrieval": retrieval_ms, "rerank": rerank_ms,
                                        "retrieval_and_rerank": retrieval_ms + rerank_ms}})
            if number % 25 == 0 or number == len(test):
                print(f"final evaluated {number}/{len(test)} queries", file=sys.stderr, flush=True)
    _assert_fingerprints(data, freeze["data_fingerprints"], "data")
    _assert_fingerprints(ROOT, freeze["source_fingerprints"], "source")
    _assert_fingerprints(output, freeze["artifact_fingerprints"], "artifact")
    fixed = {method: _summary([r["fixed_pool"][method] for r in rows], FIXED_METRICS)
             for method in ("title_bm25", "learned_pairwise")}
    corpus = {method: _summary([r["corpus"][method] for r in rows], CORPUS_METRICS)
              for method in ("fts_bm25", "fts_pairwise")}
    slices = {}
    for tag in sorted({tag for row in rows for tag in row["slices"]}):
        subset = [r for r in rows if tag in r["slices"]]
        slices[tag] = {"queries": len(subset),
                       "fixed_pool_delta": _paired(subset, "fixed_pool", "title_bm25", "learned_pairwise", ("ndcg@10",)),
                       "corpus_delta": _paired(subset, "corpus", "fts_bm25", "fts_pairwise", ("known_positive_recall@10",))}
    regression_ids = sorted((r for r in rows if r["fixed_pool"]["title_bm25"]["ndcg@10"] is not None),
                            key=lambda r: r["fixed_pool"]["learned_pairwise"]["ndcg@10"] - r["fixed_pool"]["title_bm25"]["ndcg@10"])
    report = {"schema_version": 1, "created_at_utc": _now(), "protocol": "public_cpu_v1",
              "dataset": freeze["dataset"], "synthetic": freeze["synthetic"],
              "catalog_products": len(catalog), "test_queries": len(test),
              "test_judged_pairs": sum(len(r["qrels"]) for r in test),
              "selected_epochs": obj["selected_epochs"], "freeze_sha256": file_sha256(output / "freeze.json"),
              "data_fingerprints": freeze["data_fingerprints"],
              "source_fingerprints": freeze["source_fingerprints"],
              "fixed_pool": fixed, "corpus": corpus,
              "fixed_pool_paired_delta": _paired(rows, "fixed_pool", "title_bm25", "learned_pairwise", FIXED_METRICS),
              "corpus_paired_delta": _paired(rows, "corpus", "fts_bm25", "fts_pairwise", CORPUS_METRICS),
              "slices": slices,
              "retrieval": {"pool": freeze["retrieval_pool"],
                            "empty_queries": sum(r["retrieved_count"] == 0 for r in rows),
                            "parser_rejected_queries": sum(r["retrieval_error"] is not None for r in rows),
                            "mean_candidates": statistics.fmean(r["retrieved_count"] for r in rows),
                            "candidate_coverage_definition": "macro known-positive Recall@100; reranking preserves this set"},
              "failure_analysis": {"worst_fixed_pool_delta_query_ids": [r["query_id"] for r in regression_ids[:10]],
                                   "details_file": "per_query_test.jsonl" if per_query else None,
                                   "slice_policy": "diagnostic only; no model retuning on this test"},
              "runtime": {"python": platform.python_version(), "system": platform.system(),
                          "machine": platform.machine(), "feature_load_seconds": feature_load_seconds,
                          "total_final_seconds": time.perf_counter() - started,
                          "latency": {name: _latency([r["latency_ms"][name] for r in rows])
                                      for name in ("retrieval", "rerank", "retrieval_and_rerank")},
                          "latency_protocol": "serial single process; application result cache disabled; catalog/model resident; OS cache uncontrolled; no throughput claim"},
              "metric_definitions": {"ndcg": "gain=2**grade-1; macro average; zero ideal DCG excluded",
                                     "es_positive": "grade>=2 (Exact or Substitute)",
                                     "precision": "fixed fully judged pool only; denominator=10",
                                     "corpus": "incomplete human qrels; unknown products are not labeled negatives; no corpus precision/NDCG claim",
                                     "coverage": "judged_fraction@10 measures annotation coverage, not relevance"},
              "limits": ["Text-only CPU baseline and trained linear ranker, not the historical CE/LambdaMART model",
                         "New public reproduction protocol; test overlap with prior unrelated local experiments is not excluded",
                         "Bootstrap intervals quantify query sampling variation; exploratory slices have no multiplicity correction",
                         "No online CTR/conversion, billion-product, multilingual or deep semantic accuracy claim"]}
    if per_query:
        save_jsonl(output / "per_query_test.jsonl", rows)
    save_json(output / "report.json", report)
    lines = ["# Public CPU benchmark", "", f"Catalog: {len(catalog):,}; holdout queries: {len(test)}.", "",
             "| Fixed fully judged candidates | NDCG@10 | Exact@1 | P@10 (E/S) |",
             "|---|---:|---:|---:|"]
    fmt = lambda value: "null" if value is None else f"{value:.6f}"
    for method, metrics in fixed.items():
        lines.append(f"| {method} | {fmt(metrics['ndcg@10']['mean'])} | {fmt(metrics['exact@1']['mean'])} | {fmt(metrics['p@10_es']['mean'])} |")
    lines += ["", "| Full catalog | Known-positive Recall@10 | Known-positive Recall@100 |",
              "|---|---:|---:|"]
    for method, metrics in corpus.items():
        lines.append(f"| {method} | {fmt(metrics['known_positive_recall@10']['mean'])} | {fmt(metrics['known_positive_recall@100']['mean'])} |")
    lines += ["", "Full-catalog judgments are incomplete. Recall is relative to known E/S labels; unjudged results are unknown.",
              "Paired confidence intervals, denominators, diagnostic slices and measured serial latency are in `report.json`.",
              "The model and data were frozen before test parsing. A second final run in this directory is rejected.",
              "These are newly measured public-run results, not reproduction of the historical joint-encoder scores."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "final"):
        command = sub.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data/esci-public"))
        command.add_argument("--output", type=Path, default=Path("outputs/public-cpu-v1"))
        if name == "final":
            command.add_argument("--per-query", action="store_true", help="save local query-level diagnosis; do not publish raw data")
    args = parser.parse_args(argv)
    result = select(args.data, args.output) if args.command == "select" else final(args.data, args.output, per_query=args.per_query)
    summary = {"command": args.command, "output": str(args.output), "selected_epochs": result["selected_epochs"]}
    if args.command == "final":
        summary.update(fixed_pool=result["fixed_pool"], corpus=result["corpus"],
                       fixed_pool_paired_delta=result["fixed_pool_paired_delta"],
                       corpus_paired_delta=result["corpus_paired_delta"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
