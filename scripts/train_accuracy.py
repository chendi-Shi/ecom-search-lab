"""Select using train/val only; freeze artifacts before explicitly opening test.

python scripts/train_accuracy.py select --lightgbm-site <optional site-packages>
python scripts/train_accuracy.py final
Final is resumable only with exactly the already-frozen code/config/data/model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

from ecom_search.accuracy import (ACCURACY_FEATURES, ACCURACY_SCHEMA_VERSION, RAW_CE_COLUMN,
    CompactFeatureExtractor, build_pool_features, export_trees, model_scores)
from ecom_search.cross_encoder import LocalCrossEncoder, product_text
from ecom_search.data import file_sha256, load_jsonl, save_json, save_jsonl, validate_dataset
from ecom_search.evaluate import ndcg, recall
from ecom_search.index import catalog_digest
from ecom_search.ranking import FeatureExtractor, PairwiseRanker


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hashes():
    return {str(p): file_sha256(ROOT / p) for p in (
        "ecom_search/accuracy.py", "ecom_search/cross_encoder.py", "ecom_search/ranking.py",
        "ecom_search/evaluate.py", "ecom_search/data.py", "ecom_search/index.py", "ecom_search/query.py",
        "scripts/prepare_accuracy_retrieval.py", "scripts/train_accuracy.py")}


def make_features(records, extractor, ce, output, tag):
    """Content-bound features with one complete ordered CE call per query pool."""
    import numpy as np
    binding = {"records_sha256": fingerprint(records), "catalog_sha256": catalog_digest(list(extractor.products.values())),
               "ce": ce.fingerprint, "features": ACCURACY_FEATURES,
               "ranking_code": file_sha256(ROOT / "ecom_search/ranking.py"),
               "accuracy_code": file_sha256(ROOT / "ecom_search/accuracy.py")}
    key = fingerprint(binding)
    array_file, meta_file = output / f"features-{tag}.npy", output / f"features-{tag}.json"
    if array_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta["key"] != key or meta["sha256"] != file_sha256(array_file):
            raise ValueError(f"feature cache mismatch: {tag}")
        result = np.load(array_file, allow_pickle=False)
        if result.shape != (sum(len(r["candidate_ids"]) for r in records), len(ACCURACY_FEATURES)) or not np.isfinite(result).all():
            raise ValueError("invalid feature cache shape/values")
        return result
    rows, start = [], time.perf_counter()
    for i, record in enumerate(records):
        products = [extractor.products[p] for p in record["candidate_ids"]]
        logits = ce.score_pairs([(record["query"], product_text(p)) for p in products])
        rows.extend(build_pool_features(record["query"], products, extractor, logits))
        if (i + 1) % 20 == 0 or i + 1 == len(records):
            progress = {"phase": tag, "queries": i + 1, "total": len(records),
                        "pairs": len(rows), "seconds": round(time.perf_counter() - start, 2)}
            save_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    result = np.asarray(rows, dtype=np.float64).reshape(-1, len(ACCURACY_FEATURES))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite training features")
    np.save(array_file, result, allow_pickle=False)
    save_json(meta_file, {"key": key, "binding": binding, "sha256": file_sha256(array_file)})
    return result


def metrics(records, scores):
    if any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("metric scores must be finite")
    offset, per_query = 0, []
    for r in records:
        ids = r["candidate_ids"]
        values = scores[offset:offset + len(ids)]
        if len(values) != len(ids):
            raise ValueError("score count differs from candidates")
        ranked = [p for _, p in sorted(zip(values, ids), key=lambda x: (-float(x[0]), x[1]))]
        per_query.append({"query_id": r["query_id"], "query": r["query"], "candidate_count": len(ids),
            "ndcg10": ndcg(ranked, r["qrels"], 10), "recall10": recall(ranked, r["qrels"], 10),
            "precision10": sum(r["qrels"][p] >= 2 for p in ranked[:10]) / min(10, len(ranked)),
            "exact_at1": float(r["qrels"][ranked[0]] == 3), "top10": ranked[:10]})
        offset += len(ids)
    if offset != len(scores):
        raise ValueError("excess scores")
    summary = {"queries": len(records)}
    for key in ("ndcg10", "recall10", "precision10", "exact_at1"):
        values = [r[key] for r in per_query if r[key] is not None]
        summary[key] = sum(values) / len(values) if values else None
        summary[key + "_denominator"] = len(values)
    return {"summary": summary, "per_query": per_query}


def paired_interval(a, b, key="ndcg10", repetitions=10000):
    import numpy as np
    if [r["query_id"] for r in a] != [r["query_id"] for r in b]:
        raise ValueError("bootstrap query alignment mismatch")
    diffs = np.asarray([ra[key] - rb[key] for ra, rb in zip(a, b)
                        if ra[key] is not None and rb[key] is not None])
    if not len(diffs):
        return {"queries": 0, "delta": None, "ci95": None}
    rng = np.random.default_rng(20261010)
    samples = np.asarray([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(repetitions)])
    return {"queries": len(diffs), "delta": float(diffs.mean()),
        "ci95": np.quantile(samples, [.025, .975]).tolist(), "resamples": repetitions,
        "seed": 20261010, "wins": int((diffs > 1e-12).sum()),
        "ties": int((abs(diffs) <= 1e-12).sum()), "losses": int((diffs < -1e-12).sum())}


def corpus_evaluation(records, extractor, ce, model, old, output):
    """A prespecified label-blind sample; never infer relevance of unjudged items."""
    import numpy as np
    from scripts.prepare_accuracy_retrieval import RetrievalPool
    chosen = sorted(records, key=lambda r: fingerprint([20261010, "accuracy_corpus", r["query_id"]]))[:50]
    methods = {name: [] for name in ("retrieval", "transferred_pairwise", "cross_encoder", "selected")}
    latency, total_start = [], time.perf_counter()
    pool = RetrievalPool(output / "retrieval")
    try:
        for i, record in enumerate(chosen):
            start = time.perf_counter()
            candidates = pool.search(record["query"], k=100)
            ids = [p for p, _ in candidates]
            products = [extractor.products[p] for p in ids]
            logits = ce.score_pairs([(record["query"], product_text(p)) for p in products])
            x = np.asarray(build_pool_features(record["query"], products, extractor, logits),
                           dtype=np.float64).reshape(-1, len(ACCURACY_FEATURES))
            scores = {"retrieval": [s for _, s in candidates], "transferred_pairwise": x[:, :12] @ np.asarray(old.weights),
                      "cross_encoder": logits, "selected": model_scores(model, x)}
            latency.append(1000 * (time.perf_counter() - start))
            positive = {p for p, g in record["qrels"].items() if g >= 2}
            for name, values in scores.items():
                ranked = [p for _, p in sorted(zip(values, ids), key=lambda x: (-float(x[0]), x[1]))]
                methods[name].append({"query_id": record["query_id"], "query": record["query"], "candidate_count": len(ids),
                    "known_positive_recall10": len(set(ranked[:10]) & positive) / len(positive) if positive else None,
                    "known_positive_recall100": len(set(ranked[:100]) & positive) / len(positive) if positive else None,
                    "known_positive_fraction10": len(set(ranked[:10]) & positive) / 10,
                    "judged_coverage10": sum(p in record["qrels"] for p in ranked[:10]) / 10,
                    "top10": ranked[:10]})
            if (i + 1) % 5 == 0 or i + 1 == len(chosen):
                print(json.dumps({"phase": "final_corpus", "queries": i + 1, "total": len(chosen),
                                  "seconds": round(time.perf_counter() - total_start, 2)}), flush=True)
    finally:
        pool.close()
    keys = ["known_positive_recall10", "known_positive_recall100", "known_positive_fraction10", "judged_coverage10"]
    summary = {}
    for name, rows in methods.items():
        summary[name] = {"queries": len(rows)}
        for key in keys:
            vals = [r[key] for r in rows if r[key] is not None]
            summary[name][key] = sum(vals) / len(vals) if vals else None
            summary[name][key + "_denominator"] = len(vals)
        save_jsonl(output / f"corpus-test-{name}.jsonl", rows)
    report = {"status": "completed", "queries": len(chosen), "catalog_products": len(extractor.products),
        "query_selection": "first 50 hashes SHA256([20261010, accuracy_corpus, query_id]); before any label-dependent filtering",
        "methods": summary, "selected_vs_retrieval": {key: paired_interval(methods["selected"], methods["retrieval"], key) for key in keys},
        "latency_ms": {"p50": float(np.quantile(latency, .5)), "p95": float(np.quantile(latency, .95)),
                       "scope": "serial retrieval + joint encoder + all ranker comparisons; model already loaded; content cache may hit exact repeated batches"},
        "limitations": ["Unjudged items have unknown relevance; known positive fraction is not full-catalog precision.",
                        "Same frozen candidate pool; Recall@100 must be identical after reranking.",
                        "50-query label-blind sample controls CPU budget; 200-query fully judged ranking is the primary endpoint."]}
    save_json(output / "end_to_end_report.json", report)
    return report


def run(args):
    import numpy as np
    data, out = Path(args.data).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected_path, lock_path = out / "selection.json", out / "final-lock.json"
    if args.phase == "select" and (selected_path.exists() or lock_path.exists()):
        raise ValueError("selection is frozen; choose a new development output, never tune this holdout")
    if args.phase == "final":
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        for name, sha in selected["data_files"].items():
            if file_sha256(data / name) != sha:
                raise ValueError("frozen data changed")
        if selected["code_files"] != code_hashes() or file_sha256(out / "model.json") != selected["model_sha256"]:
            raise ValueError("frozen model or evaluation code changed")
        if file_sha256(out / "protocol.json") != selected["protocol_sha256"]:
            raise ValueError("frozen evaluation protocol changed")
        if file_sha256(out / "retrieval/pool_strategy.json") != selected["retrieval_strategy_sha256"]:
            raise ValueError("frozen retrieval strategy changed")
        lock = {"selection_sha256": file_sha256(selected_path), "code_files": code_hashes()}
        if lock_path.exists():
            previous = json.loads(lock_path.read_text(encoding="utf-8"))
            if any(previous[k] != v for k, v in lock.items()):
                raise ValueError("final evaluation cannot change after opening holdout")
        else:
            save_json(lock_path, {**lock, "opened_at": timestamp()})
        if (out / "report.json").exists():
            print("Final evaluation already complete; frozen report preserved.")
            return
    catalog = load_jsonl(data / "catalog.jsonl")
    print(json.dumps({"phase": "loading_catalog", "products": len(catalog)}), flush=True)
    extractor = CompactFeatureExtractor(catalog)
    print(json.dumps({"phase": "lexical_catalog_ready", "at": timestamp()}), flush=True)
    precision = args.precision
    if args.phase == "final":
        precision = json.loads((out / "model.json").read_text(encoding="utf-8"))["cross_encoder"]["config"]["precision"]
    ce = LocalCrossEncoder(args.ce_model, cache_dir=out / "ce-cache", threads=2, max_length=128, batch_size=32, precision=precision)
    baseline = json.loads((ROOT / "outputs/ranking/model.json").read_text(encoding="utf-8"))
    old = PairwiseRanker.from_dict(baseline["ranker"])

    def comparisons(records, x, model):
        return {"bm25": metrics(records, x[:, 0]),
                "transferred_pairwise": metrics(records, x[:, :12] @ np.asarray(old.weights)),
                "cross_encoder": metrics(records, x[:, RAW_CE_COLUMN]),
                "selected": metrics(records, model_scores(model, x))}

    model_base = {"schema_version": ACCURACY_SCHEMA_VERSION, "feature_names": ACCURACY_FEATURES,
        "ranking_source_sha256": file_sha256(ROOT / "ecom_search/ranking.py"),
        "accuracy_source_sha256": file_sha256(ROOT / "ecom_search/accuracy.py"),
        "catalog_sha256": catalog_digest(catalog), "cross_encoder": {"fingerprint": ce.fingerprint,
            "max_length": 128, "config": ce.config, "document_policy": "title. Brand: brand; no description; v1"}}
    if args.phase == "select":
        if args.lightgbm_site:
            sys.path.append(str(Path(args.lightgbm_site).resolve()))
        import lightgbm as lgb
        train, val = load_jsonl(data / "train.jsonl"), load_jsonl(data / "val.jsonl")
        validate_dataset(catalog, {"train": train, "val": val})
        train = sorted(train, key=lambda r: fingerprint([20261010, "accuracy_train", r["query_id"]]))[:500]
        protocol = {"created_at": timestamp(), "train_queries": len(train), "train_query_ids": [r["query_id"] for r in train],
            "validation_queries": len(val), "selection_metric": "macro graded NDCG@10",
            "grade_gain": [0, 1, 3, 7], "relevant_grade_min": 2,
            "num_leaves": [7, 15], "iterations": [50, 100, 200, 400],
            "features": ACCURACY_FEATURES, "tie_policy": "score descending then product ID ascending",
            "feature_family": "A: original 13 features plus complete-pool CE percentile and population z-score",
            "relative_feature_policy": "Compute over every complete ordered candidate pool; preserve input order and raw CE logit at column 12. No labels enter features.",
            "cross_encoder_precision": precision,
            "end_to_end": {"holdout_query_sample": 50, "candidate_pool": 100,
                "sampling": "SHA256([20261010, accuracy_corpus, query_id]) ascending, independent of labels",
                "metrics": "known E/S Recall@10/@100 and judged coverage; unknown relevance remains unknown"},
            "candidate_policy": "same complete official judged candidates per query for every method",
            "baseline": "old frozen pairwise weights applied with new catalog-only statistics",
            "test_access": "final subcommand only, after model/config/code/file-hash freeze"}
        probe_path = out / "validation_feature_probe.json"
        if probe_path.is_file():
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            if (probe.get("status") != "completed" or probe["best_spec"]["family"] != "A"
                    or probe["best_per_family"]["A"]["feature_names"] != ACCURACY_FEATURES
                    or len(probe["grid"]) != 16
                    or probe["old_model_sha256"] != file_sha256(ROOT / "outputs/ranking/model.json")
                    or any(probe["inputs"][split]["records_sha256"] != fingerprint(records)
                           for split, records in (("train", train), ("val", val)))):
                raise ValueError("validation feature probe does not describe this development run")
            protocol["development_history"] = {
                "initial_13_feature_lambdamart_configurations": 8,
                "additional_probe_lambdamart_configurations": len(probe["grid"]),
                "total_distinct_lambdamart_configurations": 24,
                "feature_families": ["13 original features", "A: 15 features", "B: 17 features"],
                "probe_report_sha256": file_sha256(probe_path),
                "probe_script_sha256": probe["script_sha256"],
                "probe_best_spec": probe["best_spec"],
                "probe_validation_delta_vs_13_features": probe["ndcg10_delta_vs_selected_13_features"],
                "current_run": "Replay the same eight family-A configurations to export and verify the shared 15-feature implementation; no additional distinct grid points.",
                "selection_bias": "Feature families were explored after the initial validation grid. Validation gains are exploratory; only the subsequently frozen untouched holdout estimates generalization.",
            }
        save_json(out / "protocol.json", protocol)
        # Validation first provides a progress checkpoint while the longer train cache runs.
        xv = make_features(val, extractor, ce, out, "val")
        xt = make_features(train, extractor, ce, out, "train")
        labels = np.asarray([r["qrels"][p] for r in train for p in r["candidate_ids"]])
        groups = [len(r["candidate_ids"]) for r in train]
        model = {**model_base, "kind": "cross_encoder"}
        best = metrics(val, xv[:, RAW_CE_COLUMN])["summary"]["ndcg10"]
        choices = [{"kind": "cross_encoder", "ndcg10": best}]
        for leaves in protocol["num_leaves"]:
            ds = lgb.Dataset(xt, label=labels, group=groups, feature_name=ACCURACY_FEATURES, free_raw_data=False)
            params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "label_gain": [0, 1, 3, 7],
                "learning_rate": .05, "num_leaves": leaves, "min_data_in_leaf": 40,
                "lambda_l2": 1, "num_threads": 2, "deterministic": True, "force_col_wise": True,
                "verbosity": -1, "seed": 20261010, "feature_pre_filter": False}
            booster = lgb.train(params, ds, num_boost_round=400)
            for n in protocol["iterations"]:
                candidate = {**model_base, "kind": "lambdamart", "trees": export_trees(booster, n),
                             "training": {"params": params, "iterations": n, "lightgbm_version": lgb.__version__}}
                expected = booster.predict(xv, num_iteration=n)
                actual = np.asarray(model_scores(candidate, xv))
                if not np.allclose(expected, actual, rtol=1e-12, atol=1e-12):
                    raise AssertionError("exported trees do not match LightGBM")
                score = metrics(val, actual)["summary"]["ndcg10"]
                choices.append({"kind": "lambdamart", "leaves": leaves, "iterations": n, "ndcg10": score})
                print(json.dumps(choices[-1]), flush=True)
                if score > best + 1e-12:
                    model, best = candidate, score
        save_json(out / "model.json", model)
        evaluated = comparisons(val, xv, model)
        save_json(out / "validation.json", {"methods": {k: v["summary"] for k, v in evaluated.items()}, "choices": choices})
        save_json(out / "selection.json", {"frozen_at": timestamp(), "chosen_kind": model["kind"],
            "selection_metric": "macro graded NDCG@10 on 200 validation queries",
            "validation_ndcg10": best, "model_sha256": file_sha256(out / "model.json"),
            "baseline_sha256": file_sha256(ROOT / "outputs/ranking/model.json"),
            "protocol_sha256": file_sha256(out / "protocol.json"), "code_files": code_hashes(),
            "retrieval_strategy_sha256": file_sha256(out / "retrieval/pool_strategy.json"),
            "data_files": {p.name: file_sha256(p) for p in [data / f"{s}.jsonl" for s in ("catalog", "train", "val", "test")]},
            "test_labels_opened": False})
        print(json.dumps({"selected": model["kind"], "val_ndcg10": best}), flush=True)
    else:
        if selected["baseline_sha256"] != file_sha256(ROOT / "outputs/ranking/model.json"):
            raise ValueError("frozen baseline changed")
        model = json.loads((out / "model.json").read_text(encoding="utf-8"))
        if model["cross_encoder"]["fingerprint"] != ce.fingerprint:
            raise ValueError("frozen encoder changed")
        records = load_jsonl(data / "test.jsonl")
        validate_dataset(catalog, {"test": records})
        x = make_features(records, extractor, ce, out, "test")
        evaluated = comparisons(records, x, model)
        intervals = {name: {key: paired_interval(evaluated["selected"]["per_query"], result["per_query"], key)
                           for key in ("ndcg10", "precision10", "exact_at1")}
                     for name, result in evaluated.items() if name != "selected"}
        report = {"completed_at": timestamp(), "status": "completed", "protocol": json.loads((out / "protocol.json").read_text(encoding="utf-8")),
            "data": {"catalog_products": len(catalog), "test_queries": len(records), "candidate_pairs": len(x)},
            "selected_kind": model["kind"], "methods": {k: v["summary"] for k, v in evaluated.items()},
            "paired_query_bootstrap": intervals, "selection_sha256": file_sha256(selected_path),
            "limitations": ["Identical fully judged query candidate pools, not end-to-end all-catalog precision.",
                "New query holdout disjoint from old local experiment; upstream MS-MARCO pretraining overlap is unknown.",
                "ESCI English US queries with at least 20 judged products; not all traffic or online business metrics.",
                "New catalog/candidate distribution; compare methods inside this report, not old .7606 directly."]}
        for name, result in evaluated.items():
            save_jsonl(out / f"test-{name}.jsonl", result["per_query"])
        corpus_evaluation(records, extractor, ce, model, old, out)
        save_json(out / "report.json", report)
        print(json.dumps({"methods": report["methods"], "intervals": intervals}), flush=True)
    close = getattr(ce, "close", None)
    if close:
        close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "final"))
    parser.add_argument("--data", default=str(ROOT / "data/esci-v2"))
    parser.add_argument("--output", default=str(ROOT / "outputs/accuracy-v1"))
    parser.add_argument("--ce-model", default=str(ROOT / "outputs/cross-encoder-assets/ms-marco-MiniLM-L6-v2"))
    parser.add_argument("--lightgbm-site")
    parser.add_argument("--precision", choices=("float32", "dynamic_int8"), default="dynamic_int8")
    run(parser.parse_args())
