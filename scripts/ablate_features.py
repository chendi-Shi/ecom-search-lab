"""Retrospective, fixed-hyperparameter feature ablations of the public CPU model.

This is diagnosis on an already reported test set, never a new holdout claim.
The original ranker, feature schema, optimizer and candidate retrieval are reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_search.data import file_sha256, load_jsonl, save_json, save_jsonl, validate_dataset
from ecom_search.index import ProductIndex, catalog_digest
from ecom_search.ranking import FEATURE_NAMES, FeatureExtractor, PairwiseRanker
from scripts import benchmark_public as benchmark

GROUPS = {
    "title_bm25": (0,), "coverage": (1, 2), "char_trigram": (3,),
    "brand": (4, 5), "number": (6, 7), "phrase": (8, 9), "negation": (10, 11),
}
TRAINING = {"epochs": 30, "learning_rate": .05, "l2": .001, "seed": 42}
SOURCE_FILES = tuple(dict.fromkeys((*benchmark.SOURCE_FILES, "scripts/ablate_features.py")))
REFERENCE_FILES = ("model.json", "products.sqlite", "report.json")
PROTOCOL = "retrospective_feature_ablation_v1"
BOOTSTRAP_SEED = 20261013
TOLERANCE = 1e-10


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def variants():
    return {"full": (), **{"without_" + group: indices for group, indices in GROUPS.items()}}


def mask_vector(vector, indices):
    """Keep the original 12-dimensional schema and return a fresh vector."""
    if len(vector) != len(FEATURE_NAMES):
        raise ValueError("feature schema mismatch")
    return [0.0 if i in indices else value for i, value in enumerate(vector)]


class CachedExtractor:
    """Read only label-free vectors; a missing pair fails rather than recomputes."""
    names = list(FEATURE_NAMES)

    def __init__(self, products, cache, dropped=()):
        self.products, self.cache, self.dropped = products, cache, tuple(dropped)

    def features(self, query, product):
        return mask_vector(self.cache[(query, str(product["id"]))], self.dropped)


def cache_features(records, extractor):
    """Only query text and candidate IDs are read: no qrels or split labels."""
    cache = {}
    for number, row in enumerate(records, 1):
        query = benchmark._ranking_query(row["query"])
        for product_id in row["candidate_ids"]:
            key = query, str(product_id)
            if key not in cache:
                cache[key] = extractor.features(query, extractor.products[str(product_id)])
        if number % 200 == 0:
            print(f"cached {number}/{len(records)} queries", file=sys.stderr, flush=True)
    return cache


def activity(feature_rows):
    """Activity and within-query variation; neither is causal contribution."""
    def describe(indices):
        pairs = sum(len(vectors) for vectors in feature_rows)
        return {
            "pairs": pairs,
            "nonzero_pairs": sum(any(v[i] != 0 for i in indices)
                                 for vectors in feature_rows for v in vectors),
            "queries": len(feature_rows),
            "nonzero_queries": sum(any(v[i] != 0 for v in vectors for i in indices)
                                   for vectors in feature_rows),
            "varying_queries": sum(any(len({v[i] for v in vectors}) > 1 for i in indices)
                                   for vectors in feature_rows),
        }
    return {"features": {name: describe((i,)) for i, name in enumerate(FEATURE_NAMES)},
            "groups": {name: describe(indices) for name, indices in GROUPS.items()}}


def prepare(data, reference, output):
    data, reference, output = Path(data), Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("ablation output exists; never overwrite a registered run")
    manifest = benchmark._manifest(data)  # Test bytes are hashed, not decoded.
    model = read_json(reference / "model.json")
    historical = read_json(reference / "report.json")
    benchmark._assert_fingerprints(ROOT, historical["source_fingerprints"], "historical source")
    if model.get("selected_epochs") != TRAINING["epochs"]:
        raise ValueError("reference must be the already-selected 30-epoch public model")
    data_hashes = benchmark._fingerprints(data, benchmark.DATA_FILES)
    if data_hashes != model.get("data_fingerprints"):
        raise ValueError("reference data fingerprints differ")
    config = {
        "schema_version": 1, "protocol": PROTOCOL, "registered_at_utc": now(),
        "study_type": "retrospective diagnostic; test was reported before this protocol",
        "test_labels_decoded_in_this_run": False,
        "training": TRAINING, "groups": {k: list(v) for k, v in GROUPS.items()},
        "feature_names": list(FEATURE_NAMES),
        "variants": {k: list(v) for k, v in variants().items()},
        "data_fingerprints": data_hashes,
        "reference_fingerprints": benchmark._fingerprints(reference, REFERENCE_FILES),
        "source_fingerprints": benchmark._fingerprints(ROOT, SOURCE_FILES),
        "dataset": manifest.get("dataset"), "synthetic": manifest.get("synthetic"),
        "retrieval_pool": 100, "bootstrap_samples": 2000, "bootstrap_seed": BOOTSTRAP_SEED,
        "reference_tolerance": TOLERANCE,
        "selection_policy": "all variants reported; no test-dependent selection or retuning",
        "comparison_direction": "full minus ablated; positive means removal hurts",
        "multiplicity": "seven group comparisons per diagnostic and several metrics; all CIs exploratory, unadjusted",
        "attribution": "leave-one-group-out effects are conditional and non-additive; not causal/Shapley decomposition",
        "semantics": "mask feature columns only; shared tokenization, negation parsing and query planning remain unchanged",
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", config)
    return config


def check_inputs(data, reference, output):
    config = read_json(Path(output) / "protocol.json")
    if config.get("protocol") != PROTOCOL or config.get("training") != TRAINING:
        raise ValueError("unsupported or changed ablation protocol")
    if config.get("variants") != {k: list(v) for k, v in variants().items()}:
        raise ValueError("ablation variants changed")
    benchmark._assert_fingerprints(data, config["data_fingerprints"], "data")
    benchmark._assert_fingerprints(reference, config["reference_fingerprints"], "reference")
    benchmark._assert_fingerprints(ROOT, config["source_fingerprints"], "source")
    return config


def train(data, reference, output):
    data, reference, output = Path(data), Path(reference), Path(output)
    config = check_inputs(data, reference, output)
    with (output / "training_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"at_utc": now(), "protocol_sha256": file_sha256(output / "protocol.json")}, handle)
    started = time.perf_counter()
    catalog, records = load_jsonl(data / "catalog.jsonl"), load_jsonl(data / "train.jsonl")
    validate_dataset(catalog, {"train": records})
    reference_obj = read_json(reference / "model.json")
    if catalog_digest(catalog) != reference_obj["catalog_sha256"]:
        raise ValueError("catalog differs from reference")
    extractor = FeatureExtractor(catalog)
    cache_start = time.perf_counter()
    cache = cache_features(records, extractor)
    cache_seconds = time.perf_counter() - cache_start
    save_jsonl(output / "train_features.jsonl", [
        {"query": query, "product_id": pid, "features": vector}
        for (query, pid), vector in cache.items()])
    normalized = [dict(row, query=benchmark._ranking_query(row["query"])) for row in records]
    reference_model = PairwiseRanker.from_dict(reference_obj["ranker"])
    trained, reproduction = {}, None
    for name, dropped in variants().items():
        before = time.perf_counter()
        model = PairwiseRanker()
        result = model.fit(normalized, CachedExtractor(extractor.products, cache, dropped), **TRAINING)
        trained[name] = {"dropped": list(dropped), "training": result,
                         "fit_seconds": time.perf_counter() - before, "ranker": model.to_dict()}
        if name == "full":
            weight_difference = max(abs(a - b) for a, b in zip(model.weights, reference_model.weights))
            score_difference = max(abs(model.score(v) - reference_model.score(v)) for v in cache.values())
            reproduction = {"weight_max_abs_difference": weight_difference,
                            "train_score_max_abs_difference": score_difference,
                            "prediction_pairs": len(cache), "tolerance": TOLERANCE,
                            "passed": max(weight_difference, score_difference) <= TOLERANCE}
            if not reproduction["passed"]:
                save_json(output / "reproduction_failure.json", reproduction)
                raise ValueError("full reference failed reproduction; test evaluation forbidden")
        print(f"trained {name}: {trained[name]['fit_seconds']:.2f}s", file=sys.stderr, flush=True)
    check_inputs(data, reference, output)
    train_activity = activity([[cache[(row["query"], str(pid))] for pid in row["candidate_ids"]]
                               for row in normalized])
    save_json(output / "models.json", trained)
    training = {"reproduction": reproduction, "cache_pairs": len(cache), "cache_seconds": cache_seconds,
                "total_seconds": time.perf_counter() - started, "training": TRAINING,
                "activity": train_activity,
                "cost": {name: {"fit_seconds": row["fit_seconds"], "pairs": row["training"]["pairs"]}
                         for name, row in trained.items()},
                "python": platform.python_version(), "system": platform.system(),
                "cost_protocol": "serial CPU process, one shared extraction pass; fit excludes shared extraction; OS load uncontrolled"}
    save_json(output / "training.json", training)
    save_json(output / "models_frozen.json", {
        "frozen_at_utc": now(), "test_labels_decoded_in_this_run": False,
        "artifact_fingerprints": benchmark._fingerprints(output, (
            "protocol.json", "models.json", "training.json", "train_features.jsonl"))})
    return training


def delta_rows(rows, section, ablated, metric, group=None):
    subset = rows if group is None else [row for row in rows if row["varying_groups"][section][group]]
    return benchmark.paired_bootstrap(
        [row[section][ablated][metric] for row in subset],
        [row[section]["full"][metric] for row in subset], samples=2000, seed=BOOTSTRAP_SEED)


def evaluate(data, reference, output):
    data, reference, output = Path(data), Path(reference), Path(output)
    config = check_inputs(data, reference, output)
    frozen = read_json(output / "models_frozen.json")
    benchmark._assert_fingerprints(output, frozen["artifact_fingerprints"], "trained artifact")
    with (output / "evaluation_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"at_utc": now(), "freeze_sha256": file_sha256(output / "models_frozen.json")}, handle)
    started = time.perf_counter()
    catalog, test = load_jsonl(data / "catalog.jsonl"), load_jsonl(data / "test.jsonl")
    validate_dataset(catalog, {"test": test})
    for row in test:
        full = row.get("full_qrels", row["qrels"])
        if any(full.get(pid) != grade for pid, grade in row["qrels"].items()):
            raise ValueError("full_qrels must preserve all fixed-pool judgments")
    extractor = FeatureExtractor(catalog)
    model_objects = read_json(output / "models.json")
    models = {name: PairwiseRanker.from_dict(row["ranker"]) for name, row in model_objects.items()}
    reference_model = PairwiseRanker.from_dict(read_json(reference / "model.json")["ranker"])
    all_variants = list(models) + ["sensitivity_without_" + name for name in GROUPS]
    evaluation_rows, pools, feature_sets = [], [], {"fixed_pool": [], "corpus": []}
    max_test_score_difference = 0.0
    with ProductIndex(reference / "products.sqlite", cache_size=0) as index:
        for number, row in enumerate(test, 1):
            query = benchmark._ranking_query(row["query"])
            retrieval_error = None
            try:
                hits = index.search(row["query"], k=config["retrieval_pool"])
            except ValueError as exc:
                hits, retrieval_error = [], str(exc)
            corpus_ids, fixed_ids = [str(p["id"]) for p in hits], list(row["candidate_ids"])
            union_ids = list(dict.fromkeys(fixed_ids + corpus_ids))
            vectors = {pid: extractor.features(query, extractor.products[pid]) for pid in union_ids}
            max_test_score_difference = max(max_test_score_difference, max(
                (abs(models["full"].score(v) - reference_model.score(v)) for v in vectors.values()), default=0))
            pools.append({"query_id": row["query_id"], "fixed_ids": fixed_ids,
                          "retrieved_ids": corpus_ids, "retrieval_error": retrieval_error})
            record = {"query_id": row["query_id"], "fixed_pool": {}, "corpus": {},
                      "varying_groups": {}, "retrieval": benchmark._corpus_metrics(corpus_ids, row.get("full_qrels", row["qrels"]))}
            for section, ids, qrels, metrics in (
                ("fixed_pool", fixed_ids, row["qrels"], benchmark._fixed_metrics),
                ("corpus", corpus_ids, row.get("full_qrels", row["qrels"]), benchmark._corpus_metrics)):
                features = [vectors[pid] for pid in ids]
                feature_sets[section].append(features)
                record["varying_groups"][section] = {
                    name: any(len({v[i] for v in features}) > 1 for i in indices)
                    for name, indices in GROUPS.items()}
                for name in all_variants:
                    if name.startswith("sensitivity_without_"):
                        dropped = GROUPS[name.removeprefix("sensitivity_without_")]
                        model = reference_model
                    else:
                        dropped, model = tuple(model_objects[name]["dropped"]), models[name]
                    scores = [model.score(mask_vector(v, dropped)) for v in features]
                    record[section][name] = metrics(benchmark._rank(ids, scores), qrels)
            evaluation_rows.append(record)
            if number % 25 == 0:
                print(f"evaluated {number}/{len(test)} historical test queries", file=sys.stderr, flush=True)
    if max_test_score_difference > TOLERANCE:
        raise ValueError("test predictions of full reference differ; no report written")
    save_jsonl(output / "candidate_pools.jsonl", pools)
    save_jsonl(output / "per_query.jsonl", evaluation_rows)
    summaries = {section: {name: benchmark._summary([r[section][name] for r in evaluation_rows], metrics)
                           for name in all_variants}
                 for section, metrics in (("fixed_pool", benchmark.FIXED_METRICS), ("corpus", benchmark.CORPUS_METRICS))}
    historical = read_json(reference / "report.json")
    # Reproduce all published macro metrics before reporting any ablation.
    reproduction_metrics = {}
    for section, historical_name in (("fixed_pool", "learned_pairwise"), ("corpus", "fts_pairwise")):
        for metric, value in summaries[section]["full"].items():
            old = historical[section][historical_name][metric]
            difference = abs(value["mean"] - old["mean"]) if value["mean"] is not None and old["mean"] is not None else 0
            if value["queries"] != old["queries"] or (value["mean"] is None) != (old["mean"] is None) or difference > TOLERANCE:
                raise ValueError(f"historical full metrics differ: {section}/{metric}")
            reproduction_metrics[section + "/" + metric] = difference
    comparisons = {}
    for diagnostic, prefix in (("retrained", "without_"), ("inference_sensitivity", "sensitivity_without_")):
        comparisons[diagnostic] = {}
        for group in GROUPS:
            comparisons[diagnostic][group] = {
                section: {metric: delta_rows(evaluation_rows, section, prefix + group, metric) for metric in metrics}
                for section, metrics in (("fixed_pool", benchmark.FIXED_METRICS), ("corpus", benchmark.CORPUS_METRICS))}
            comparisons[diagnostic][group]["supported_fixed_ndcg"] = delta_rows(
                evaluation_rows, "fixed_pool", prefix + group, "ndcg@10", group=group)
    check_inputs(data, reference, output)
    benchmark._assert_fingerprints(output, frozen["artifact_fingerprints"], "trained artifact")
    report = {"schema_version": 1, "protocol": PROTOCOL, "created_at_utc": now(),
              "study_type": config["study_type"], "catalog_products": len(catalog), "test_queries": len(test),
              "test_pairs": sum(len(row["candidate_ids"]) for row in test),
              "data_fingerprints": config["data_fingerprints"], "source_fingerprints": config["source_fingerprints"],
              "protocol_sha256": file_sha256(output / "protocol.json"),
              "models_frozen_sha256": file_sha256(output / "models_frozen.json"),
              "candidate_pool_sha256": file_sha256(output / "candidate_pools.jsonl"),
              "candidate_policy": "one retrieval per query; identical saved 100-candidate pool and tie-break across all 15 variants",
              "candidate_count_mean": statistics.fmean(len(p["retrieved_ids"]) for p in pools),
              "training": read_json(output / "training.json"),
              "reference_reproduction": {"passed": True, "test_score_max_abs_difference": max_test_score_difference,
                                         "published_metric_abs_differences": reproduction_metrics},
              "metrics": summaries, "comparisons_full_minus_ablated": comparisons,
              "retrieval_baseline": benchmark._summary([r["retrieval"] for r in evaluation_rows], benchmark.CORPUS_METRICS),
              "activity": {section: activity(feature_sets[section]) for section in feature_sets},
              "evaluation_seconds": time.perf_counter() - started,
              "interpretation": [config["comparison_direction"], config["multiplicity"], config["attribution"], config["semantics"],
                                 "Inference sensitivity zeros frozen-model inputs without retraining; retraining lets remaining features compensate",
                                 "Activity requires observed feature support; varying-query slices are descriptive and not an independent test",
                                 "All corpus judgments are incomplete; Recall@100 is unchanged by reranking the same set",
                                 "Do not promote an ablated winner or claim unseen-test generalization from this retrospective study"]}
    save_json(output / "report.json", report)
    write_markdown(report, output / "report.md")
    return report


def write_markdown(report, path):
    lines = ["# Retrospective feature ablation", "",
             "Already-reported public test; fixed hyperparameters; all variants reported. Positive delta means full minus removed.", "",
             "| Removed group | Retrained NDCG@10 | Full-minus-removed | Query bootstrap 95% CI | Frozen-input sensitivity delta | Varying test queries |",
             "|---|---:|---:|---|---:|---:|"]
    comparisons = report["comparisons_full_minus_ablated"]
    for group in GROUPS:
        value = report["metrics"]["fixed_pool"]["without_" + group]["ndcg@10"]["mean"]
        delta = comparisons["retrained"][group]["fixed_pool"]["ndcg@10"]
        sensitivity = comparisons["inference_sensitivity"][group]["fixed_pool"]["ndcg@10"]["delta"]
        active = report["activity"]["fixed_pool"]["groups"][group]["varying_queries"]
        lines.append(f"| {group} | {value:.6f} | {delta['delta']:+.6f} | [{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}] | {sensitivity:+.6f} | {active} |")
    lines += ["", f"Full reference NDCG@10: {report['metrics']['fixed_pool']['full']['ndcg@10']['mean']:.6f}.", "",
              "These effects overlap and cannot be summed. CIs are exploratory and unadjusted across groups/metrics.",
              "Removing a feature group is not removing shared query parsing or the entire feature engineering subsystem.",
              "Full-catalog known-positive recall, activity denominators, reproduction checks and measured costs are in report.json.",
              "The description column may be inactive in title/brand-only input; a zero effect then says nothing about richer descriptions."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "run"))
    parser.add_argument("--data", type=Path, default=Path("data/esci-public"))
    parser.add_argument("--reference", type=Path, default=Path("outputs/public-cpu-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation-features-v1"))
    args = parser.parse_args(argv)
    stages = (prepare, train, evaluate) if args.command == "run" else (globals()[args.command],)
    for stage in stages:
        stage(args.data, args.reference, args.output)
        print(json.dumps({"completed": stage.__name__, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
