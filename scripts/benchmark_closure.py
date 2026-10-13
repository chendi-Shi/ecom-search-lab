"""Fresh-query, validation-selected retrieval repair with a conservative release gate.

prepare -> train -> select -> final. Each stage binds immutable artifacts. Test
labels are parsed only by final, which acquires an exclusive one-attempt marker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecom_search.closure import ClosureEngine, ranking_query
from ecom_search.data import file_sha256, load_jsonl, normalized_query, validate_dataset
from ecom_search.index import build_index, catalog_digest
from ecom_search.ranking import FEATURE_NAMES, PairwiseRanker
from scripts import benchmark_public as old

PROTOCOL = "fresh_query_closure_v1"
TRAINING = {"epochs": 30, "learning_rate": .05, "l2": .001, "seed": 20261013}
POOL = 100
BASELINE = "legacy_bm25"
VARIANTS = {f"{p}_{r}": {"policy": p, "ranker": r}
            for p in ("legacy", "soft") for r in ("bm25", "judged", "retrieved")}
METRICS = ("known_positive_recall@10", "candidate_recall@100", "judged_fraction@10")
PRIMARY, COVERAGE = METRICS[:2]
GATE = {"primary_ci95_lower_strictly_above": 0.0,
        "candidate_recall_delta_ci95_lower_at_least": -.01}
BOOTSTRAP = {"samples": 2000, "seed": 20261013}
TRAINED_FILES = ("protocol.json", "models.json", "training.json", "products.sqlite")
SELECTED_FILES = ("protocol.json", "trained.json", "models.json", "training.json",
                  "products.sqlite", "validation.json")
SOURCES = tuple(dict.fromkeys((*old.SOURCE_FILES, "ecom_search/closure.py",
    "scripts/prepare_closure_esci.py", "scripts/benchmark_closure.py")))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def marker(output, name, bindings):
    with (Path(output) / name).open("x", encoding="utf-8") as handle:
        json.dump({"at_utc": now(), **bindings}, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def fingerprints(directory, names):
    return old._fingerprints(directory, names)


def check(data, output):
    config = read(Path(output) / "protocol.json")
    if config.get("protocol") != PROTOCOL or config.get("training") != TRAINING:
        raise ValueError("unsupported/changed protocol")
    if config.get("variants") != VARIANTS or config.get("pool") != POOL:
        raise ValueError("changed variants or pool")
    if any(config.get(k) != v for k, v in {"baseline": BASELINE, "primary": PRIMARY,
            "promotion_rule": GATE, "bootstrap": BOOTSTRAP}.items()):
        raise ValueError("changed selection or release rules")
    if set(config["data_fingerprints"]) != set((*old.DATA_FILES, "selection.json")) or set(config["source_fingerprints"]) != set(SOURCES):
        raise ValueError("required input fingerprints missing or unexpected")
    old._assert_fingerprints(data, config["data_fingerprints"], "data")
    old._assert_fingerprints(ROOT, config["source_fingerprints"], "source")
    return config


def prepare(data, output):
    data, output = Path(data), Path(output)
    if output.exists():
        raise FileExistsError("choose a new run directory; never erase run markers")
    manifest = old._manifest(data)  # Hash test bytes, do not decode test labels.
    exclusion = manifest.get("previous_experiment_exclusion", {})
    if (manifest.get("protocol") != "closure-esci-v1" or manifest.get("selection_uses_esci_label") is not False
            or not exclusion.get("registry_content_sha256")
            or exclusion.get("selected_overlap_by_id") != 0
            or exclusion.get("selected_overlap_by_normalized_query") != 0):
        raise ValueError("a grade-blind fresh-query cohort with registered historical exclusions is required")
    old._assert_fingerprints(ROOT, {name: info["sha256"] for name, info in manifest["source_files"].items()}, "preparation source")
    config = {"protocol": PROTOCOL, "registered_at_utc": now(), "training": TRAINING,
        "pool": POOL, "variants": VARIANTS, "baseline": BASELINE,
        "primary": PRIMARY, "positive_grades": [2, 3],
        "test_labels_decoded": False,
        "previous_experiment_exclusion": exclusion,
        "data_fingerprints": fingerprints(data, (*old.DATA_FILES, "selection.json")),
        "source_fingerprints": fingerprints(ROOT, SOURCES),
        "dataset": manifest.get("dataset"), "synthetic": manifest.get("synthetic"),
        "selection_rule": "highest validation macro R@10 among candidate R@100 >= baseline mean; ties follow variant order, baseline first",
        "promotion_rule": GATE,
        "bootstrap": BOOTSTRAP,
        "test_comparisons": "one frozen selected vs baseline primary comparison; coverage is a conjunctive safety guard",
        "training_labels": "retrieved model uses soft top100 intersect existing human qrels; unjudged is unknown, never a negative",
        "population": "all selected queries retained; no-positive queries have undefined recall and are counted separately",
        "scope": "fresh query IDs/text vs three historical cohorts; finite judged-union catalog; products may overlap history; incomplete judgments",
        "capacity_note": "identity-only scan left 1080 eligible official-train queries; use 800 train/200 val/300 test, min20 candidates"}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "protocol.json", config)
    return config


class CachedExtractor:
    names = FEATURE_NAMES

    def __init__(self, products, cache):
        self.products, self.cache = products, cache

    def features(self, query, product):
        return self.cache[(query, product["id"])]


def training_records(records, engine):
    judged, retrieved, coverage = [], [], []
    for i, row in enumerate(records, 1):
        query = ranking_query(row["query"])
        hits = engine.retrieve(row["query"], policy="soft", pool=POOL)
        ids = [hit["id"] for hit in hits if hit["id"] in row["qrels"]]
        judged.append(dict(row, query=query))
        retrieved.append(dict(row, query=query, candidate_ids=ids))
        grades = {row["qrels"][pid] for pid in ids}
        coverage.append({"retrieved": len(hits), "judged": len(ids),
                         "has_pairs": len(grades) > 1})
        if i % 100 == 0:
            print(f"retrieved training {i}/{len(records)}", flush=True)
    return judged, retrieved, {"queries": len(records),
        "retrieved_products": sum(r["retrieved"] for r in coverage),
        "judged_products": sum(r["judged"] for r in coverage),
        "unjudged_ignored": sum(r["retrieved"] - r["judged"] for r in coverage),
        "queries_with_pairs": sum(r["has_pairs"] for r in coverage),
        "queries_without_pairs": sum(not r["has_pairs"] for r in coverage),
        "queries_with_no_judged_hits": sum(r["judged"] == 0 for r in coverage)}


def train(data, output):
    data, output = Path(data), Path(output)
    check(data, output)
    marker(output, "training_started.json", {"protocol_sha256": file_sha256(output / "protocol.json")})
    started = time.perf_counter()
    catalog, records = load_jsonl(data / "catalog.jsonl"), load_jsonl(data / "train.jsonl")
    validate_dataset(catalog, {"train": records})
    info = build_index(catalog, output / "products.sqlite")
    info.pop("path", None)
    models, costs = {}, {}
    with ClosureEngine(data / "catalog.jsonl", output / "products.sqlite") as engine:
        judged, retrieved, coverage = training_records(records, engine)
        cache = {}
        for i, row in enumerate(judged, 1):
            for pid in row["candidate_ids"]:
                cache[(row["query"], pid)] = engine.extractor.features(row["query"], engine.extractor.products[pid])
            if i % 100 == 0:
                print(f"features {i}/{len(judged)}", flush=True)
        for name, rows in (("judged", judged), ("retrieved", retrieved)):
            before = time.perf_counter()
            model = PairwiseRanker()
            fit = model.fit(rows, CachedExtractor(engine.extractor.products, cache), **TRAINING)
            models[name] = model.to_dict()
            costs[name] = {"fit": fit, "seconds": time.perf_counter() - before}
            # Durable per-model checkpoints preserve work after interruption.
            atomic_json(output / (name + "_checkpoint.json"), {"model": models[name], "cost": costs[name]})
            print(f"trained {name}: {fit['pairs']} pairs", flush=True)
    check(data, output)
    atomic_json(output / "models.json", models)
    atomic_json(output / "training.json", {"training": TRAINING, "models": costs,
        "retrieved_coverage": coverage, "feature_pairs": len(cache), "index": info,
        "total_seconds": time.perf_counter() - started, "python": platform.python_version(),
        "system": platform.system(), "cost_scope": "single CPU process; uncontrolled host load"})
    atomic_json(output / "trained.json", {"protocol": PROTOCOL, "frozen_at_utc": now(),
        "test_labels_decoded": False, "catalog_sha256": catalog_digest(catalog),
        "artifacts": fingerprints(output, TRAINED_FILES)})
    return read(output / "training.json")


def verify_trained(data, output):
    config = check(data, output)
    trained = read(Path(output) / "trained.json")
    if trained.get("protocol") != PROTOCOL or set(trained.get("artifacts", {})) != set(TRAINED_FILES):
        raise ValueError("invalid training freeze or missing artifact fingerprints")
    old._assert_fingerprints(output, trained["artifacts"], "training artifact")
    if catalog_digest(load_jsonl(Path(data) / "catalog.jsonl")) != trained.get("catalog_sha256"):
        raise ValueError("trained catalog fingerprint differs")
    return config, trained


def metrics(result, qrels):
    ids = [r["id"] for r in result["ranking"]]
    return {PRIMARY: old.recall(ids, qrels, 10),
        COVERAGE: old.recall(result["candidate_ids"], qrels, POOL),
        "judged_fraction@10": sum(pid in qrels for pid in ids[:10]) / 10}


def evaluate_rows(records, engine, variants):
    rows, transcripts, latency = [], [], {name: [] for name in variants}
    for i, row in enumerate(records, 1):
        scores = {}
        for name in variants:
            before = time.perf_counter()
            result = engine.search(row["query"], **VARIANTS[name], pool=POOL, k=10)
            latency[name].append((time.perf_counter() - before) * 1000)
            scores[name] = metrics(result, row["qrels"])
            transcripts.append({"query_id": row["query_id"], "variant": name,
                "candidate_ids": result["candidate_ids"], "ranking": result["ranking"]})
        rows.append({"query_id": row["query_id"], "metrics": scores})
        if i % 50 == 0:
            print(f"evaluated {i}/{len(records)}", flush=True)
    return rows, transcripts, {name: old._latency(values) for name, values in latency.items()}


def summarize(rows, variants):
    return {name: old._summary([row["metrics"][name] for row in rows], METRICS) for name in variants}


def select_variant(summary):
    baseline_coverage = summary[BASELINE][COVERAGE]["mean"]
    if baseline_coverage is None or summary[BASELINE][PRIMARY]["mean"] is None:
        raise ValueError("validation contains no queries with known positives")
    eligible = [name for name in VARIANTS if summary[name][COVERAGE]["mean"] >= baseline_coverage]
    # Stable max resolves exact ties using insertion order, baseline first.
    return max(eligible, key=lambda name: summary[name][PRIMARY]["mean"]), eligible


def select(data, output):
    data, output = Path(data), Path(output)
    _, trained = verify_trained(data, output)
    marker(output, "selection_started.json", {"trained_sha256": file_sha256(output / "trained.json")})
    catalog, records = load_jsonl(data / "catalog.jsonl"), load_jsonl(data / "val.jsonl")
    train_rows = load_jsonl(data / "train.jsonl")
    validate_dataset(catalog, {"train": train_rows, "val": records})
    models = {name: value["weights"] for name, value in read(output / "models.json").items()}
    with ClosureEngine(data / "catalog.jsonl", output / "products.sqlite", models=models) as engine:
        rows, transcripts, latency = evaluate_rows(records, engine, VARIANTS)
    summary = summarize(rows, VARIANTS)
    selected, eligible = select_variant(summary)
    atomic_json(output / "validation.json", {"queries": len(records), "summary": summary,
        "selected": selected, "eligible": eligible, "latency": latency})
    atomic_json(output / "validation_per_query.json", rows)
    atomic_json(output / "validation_transcripts.json", transcripts)
    verify_trained(data, output)
    atomic_json(output / "selection.json", {"protocol": PROTOCOL, "frozen_at_utc": now(),
        "test_labels_decoded": False, "selected": selected, "baseline": BASELINE,
        "pool": POOL, "variant": VARIANTS[selected], "catalog_sha256": trained["catalog_sha256"],
        "train_val_query_ids": sorted(row["query_id"] for row in train_rows + records),
        "train_val_query_hashes": sorted(hashlib.sha256(normalized_query(row["query"]).encode()).hexdigest() for row in train_rows + records),
        "artifacts": fingerprints(output, SELECTED_FILES)})
    return read(output / "validation.json")


def verify_selection(data, output):
    config, trained = verify_trained(data, output)
    selection = read(Path(output) / "selection.json")
    if set(selection.get("artifacts", {})) != set(SELECTED_FILES):
        raise ValueError("selection artifact fingerprints missing or unexpected")
    old._assert_fingerprints(output, selection["artifacts"], "selection artifact")
    validation = read(Path(output) / "validation.json")
    selected, eligible = select_variant(validation["summary"])
    expected = {"protocol": PROTOCOL, "selected": selected, "baseline": BASELINE,
                "pool": POOL, "variant": VARIANTS[selected], "test_labels_decoded": False,
                "catalog_sha256": trained["catalog_sha256"]}
    if any(selection.get(k) != v for k, v in expected.items()) or validation.get("selected") != selected or validation.get("eligible") != eligible:
        raise ValueError("selection disagrees with validation winner or fixed policy")
    return config, selection


def compare(rows, selected, metric):
    result = old.paired_bootstrap([row["metrics"][BASELINE][metric] for row in rows],
        [row["metrics"][selected][metric] for row in rows], **BOOTSTRAP)
    result["method"] = "paired query percentile bootstrap; selected on separate validation; primary single comparison, coverage conjunctive gate"
    return result


def release_decision(selected, primary, coverage):
    reasons = []
    if selected == BASELINE:
        reasons.append("validation_selected_baseline")
    if primary["ci95"] is None or primary["ci95"][0] <= GATE["primary_ci95_lower_strictly_above"]:
        reasons.append("primary_lower_bound_not_above_zero")
    if coverage["ci95"] is None or coverage["ci95"][0] < GATE["candidate_recall_delta_ci95_lower_at_least"]:
        reasons.append("candidate_recall_noninferiority_not_demonstrated")
    return {"promoted": not reasons, "serving_variant": selected if not reasons else BASELINE,
            "selected_variant": selected, "reasons": reasons}


def final(data, output):
    data, output = Path(data), Path(output)
    config, selection = verify_selection(data, output)
    selected = selection["selected"]
    selection_sha = file_sha256(output / "selection.json")
    marker(output, "test_started.json", {"selection_sha256": selection_sha})
    catalog, records = load_jsonl(data / "catalog.jsonl"), load_jsonl(data / "test.jsonl")
    validate_dataset(catalog, {"test": records})
    seen_ids, seen_queries = set(selection["train_val_query_ids"]), set(selection["train_val_query_hashes"])
    if any(row["query_id"] in seen_ids or hashlib.sha256(normalized_query(row["query"]).encode()).hexdigest() in seen_queries for row in records):
        raise ValueError("test identity overlaps train/validation")
    variants = list(dict.fromkeys((BASELINE, selected)))
    models = {name: value["weights"] for name, value in read(output / "models.json").items()}
    with ClosureEngine(data / "catalog.jsonl", output / "products.sqlite", models=models) as engine:
        rows, transcripts, latency = evaluate_rows(records, engine, variants)
    primary, coverage = compare(rows, selected, PRIMARY), compare(rows, selected, COVERAGE)
    decision = release_decision(selected, primary, coverage)
    verify_trained(data, output)
    old._assert_fingerprints(output, selection["artifacts"], "selection artifact")
    if file_sha256(output / "selection.json") != selection_sha:
        raise ValueError("selection changed during test evaluation")
    atomic_json(output / "test_per_query.json", rows)
    atomic_json(output / "test_transcripts.json", transcripts)
    report = {"protocol": PROTOCOL, "completed_at_utc": now(), "queries": len(records),
        "products": len(catalog), "judged_pairs": sum(len(row["qrels"]) for row in records),
        "queries_without_known_positives": sum(not any(g >= 2 for g in row["qrels"].values()) for row in records),
        "summary": summarize(rows, variants), "primary_comparison": primary,
        "candidate_coverage_comparison": coverage, "decision": decision,
        "latency": latency, "latency_note": "sequential CPU diagnostic, baseline then selected; lexical LRU disabled by shared engine; uncontrolled host load",
        "validation": read(output / "validation.json"), "data_fingerprints": config["data_fingerprints"],
        "source_fingerprints": config["source_fingerprints"],
        "protocol_sha256": file_sha256(output / "protocol.json"),
        "selection_sha256": selection_sha,
        "prediction_sha256": file_sha256(output / "test_transcripts.json"),
        "scope": config["scope"], "no_test_retuning": True}
    atomic_json(output / "report.json", report)
    atomic_json(output / "release.json", {"protocol": PROTOCOL, "created_at_utc": now(),
        **decision, "pool": POOL, "variant": VARIANTS[decision["serving_variant"]],
        "catalog_sha256": catalog_digest(catalog), "data_fingerprints": config["data_fingerprints"],
        "source_fingerprints": config["source_fingerprints"],
        "artifacts": fingerprints(output, ("protocol.json", "trained.json", "models.json", "training.json",
            "selection.json", "validation.json", "products.sqlite", "report.json"))})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "select", "final"))
    parser.add_argument("--data", type=Path, default=Path("data/esci-closure-v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/closure-v1"))
    args = parser.parse_args()
    result = globals()[args.stage](args.data, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
