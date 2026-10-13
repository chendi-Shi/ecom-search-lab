"""Retrospective, fixed-hyperparameter ablations of the frozen 15-feature ranker.

Freeze the implementation/input hashes before run; never select a new model from
these already-seen test outcomes. Only aggregate reports are safe to publish.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecom_search.accuracy import ACCURACY_FEATURES, CompactFeatureExtractor, complete_pool_features, model_scores
from ecom_search.cross_encoder import LocalCrossEncoder
from ecom_search.data import file_sha256, load_jsonl, save_json, validate_dataset
from ecom_search.index import catalog_digest
from scripts.train_accuracy import fingerprint, make_features, metrics

SEED = 20261010
ITERATIONS = 200
RESAMPLES = 50000
PARAMS = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
          "label_gain": [0, 1, 3, 7], "learning_rate": .05, "num_leaves": 7,
          "min_data_in_leaf": 40, "lambda_l2": 1, "num_threads": 2,
          "deterministic": True, "force_col_wise": True, "verbosity": -1,
          "seed": SEED, "feature_pre_filter": False}
GROUPS = {"bm25": [0], "coverage": [1, 2], "char_trigram": [3],
          "brand": [4, 5], "number": [6, 7], "phrase_bigram": [8, 9],
          "negation": [10, 11], "ce_all": [12, 13, 14], "pool_stats": [13, 14],
          "ce_raw": [12], "percentile": [13], "zscore": [14]}
VARIANTS = {"full15": list(range(15)), **{
    "without_" + name: [i for i in range(15) if i not in columns]
    for name, columns in GROUPS.items()}}
# Removing all CE columns is exactly lexical12. Removing both relative columns
# is exactly lexical+CE13. Do not train duplicate variants under different names.
CONTRASTS = {"add_ce_to_lexical": ("without_pool_stats", "without_ce_all"),
             **{"drop_" + name: ("full15", "without_" + name) for name in GROUPS}}
METRICS = ("ndcg10", "precision10", "exact_at1", "recall10")
SOURCE_FILES = ("scripts/ablate_neural.py", "scripts/train_accuracy.py",
                "ecom_search/accuracy.py", "ecom_search/cross_encoder.py",
                "ecom_search/ranking.py", "ecom_search/evaluate.py",
                "ecom_search/data.py", "ecom_search/index.py", "ecom_search/query.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_lightgbm(site=None):
    """Select only the explicit LightGBM package, retaining this Python's deps."""
    if site:
        if "lightgbm" in sys.modules:
            raise RuntimeError("explicit LightGBM package must be chosen before import")
        package = Path(site).resolve() / "lightgbm"
        spec = importlib.util.spec_from_file_location("lightgbm", package / "__init__.py",
                                                     submodule_search_locations=[str(package)])
        if spec is None or spec.loader is None:
            raise ImportError("explicit LightGBM package not found")
        module = importlib.util.module_from_spec(spec)
        sys.modules["lightgbm"] = module
        spec.loader.exec_module(module)
        return module
    import lightgbm
    return lightgbm


def ensure_selection(reference, data):
    selected = read_json(reference / "selection.json")
    for name, expected in selected["data_files"].items():
        if file_sha256(data / name) != expected:
            raise ValueError("historical dataset fingerprint mismatch: " + name)
    for name in ("model", "protocol"):
        if file_sha256(reference / (name + ".json")) != selected[name + "_sha256"]:
            raise ValueError("historical artifact fingerprint mismatch: " + name)
    for name, expected in selected["code_files"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("historical implementation changed: " + name)
    model = read_json(reference / "model.json")
    if model["training"]["params"] != PARAMS or model["training"]["iterations"] != ITERATIONS:
        raise ValueError("reference is not the fixed 7-leaf/200-round model")
    return selected, model


def protocol_binding(args):
    data, reference = Path(args.data), Path(args.reference)
    selected, model = ensure_selection(reference, data)
    cache = {}
    if args.feature_cache:
        for split in ("train", "val", "test"):
            for suffix in ("json", "npy"):
                name = f"features-{split}.{suffix}"
                cache[name] = file_sha256(Path(args.feature_cache) / name)
    return {"schema_version": 1, "status": "retrospective_diagnostic",
            "test_previously_seen": True, "new_model_selection": False,
            "params": PARAMS, "iterations": ITERATIONS, "variants": VARIANTS,
            "groups": GROUPS, "contrasts": CONTRASTS, "metrics": list(METRICS),
            "train_queries": 500, "train_selection": "SHA256([20261010, accuracy_train, query_id]) ascending",
            "candidate_policy": "Every original judged candidate, identical order and pools across variants",
            "tie_policy": "score descending, product ID ascending",
            "bootstrap": {"unit": "query", "seed": SEED, "resamples": RESAMPLES,
                "ndcg_family_tests": len(CONTRASTS), "familywise_alpha": .05,
                "method": "paired percentile bootstrap; Bonferroni interval for the prespecified NDCG contrast family"},
            "historical_model_sha256": selected["model_sha256"],
            "historical_selection_sha256": file_sha256(reference / "selection.json"),
            "historical_report_sha256": file_sha256(reference / "report.json"),
            "historical_protocol_sha256": selected["protocol_sha256"],
            "data_files": selected["data_files"], "feature_cache_files": cache,
            "encoder_fingerprint": model["cross_encoder"]["fingerprint"],
            "lightgbm_version": model["training"]["lightgbm_version"],
            "source_files": {name: file_sha256(ROOT / name) for name in SOURCE_FILES}}


def verified_features(records, extractor, ce, output, split, cache=None):
    """Refuse stale/reordered/tampered arrays; never silently trust a cache name."""
    import numpy as np
    if cache is None:
        return make_features(records, extractor, ce, output, split)
    array = Path(cache) / f"features-{split}.npy"
    meta = read_json(Path(cache) / f"features-{split}.json")
    binding = {"records_sha256": fingerprint(records),
               "catalog_sha256": catalog_digest(list(extractor.products.values())),
               "ce": ce.fingerprint, "features": ACCURACY_FEATURES,
               "ranking_code": file_sha256(ROOT / "ecom_search/ranking.py"),
               "accuracy_code": file_sha256(ROOT / "ecom_search/accuracy.py")}
    if meta.get("binding") != binding or meta.get("key") != fingerprint(binding):
        raise ValueError("feature cache binding mismatch: " + split)
    if meta.get("sha256") != file_sha256(array):
        raise ValueError("feature cache byte checksum mismatch: " + split)
    rows = np.load(array, allow_pickle=False)
    expected_shape = (sum(len(r["candidate_ids"]) for r in records), len(ACCURACY_FEATURES))
    if rows.shape != expected_shape or rows.dtype != np.float64 or not np.isfinite(rows).all():
        raise ValueError("feature cache shape, dtype or finiteness mismatch")
    offset = 0
    for record in records:
        count = len(record["candidate_ids"])
        pool = rows[offset:offset + count]
        recomputed = np.asarray(complete_pool_features(pool[:, :13]))
        if not np.array_equal(pool, recomputed):
            raise ValueError("pool-relative feature computation mismatch")
        offset += count
    return rows


def paired_contrast(left, right, key, *, family_tests=1, repetitions=RESAMPLES):
    import numpy as np
    if [r["query_id"] for r in left] != [r["query_id"] for r in right]:
        raise ValueError("paired query identity/order mismatch")
    if family_tests < 1 or repetitions < 1:
        raise ValueError("invalid inference settings")
    differences = np.asarray([a[key] - b[key] for a, b in zip(left, right)
                              if a[key] is not None and b[key] is not None], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError("nonfinite metric")
    if len(differences) == 0:
        return {"queries": 0, "delta": None, "ci95": None, "familywise_ci": None}
    rng = np.random.default_rng(SEED)
    means = []
    for begin in range(0, repetitions, 1000):
        indices = rng.integers(len(differences), size=(min(1000, repetitions - begin), len(differences)))
        means.extend(differences[indices].mean(axis=1).tolist())
    alpha = .05 / family_tests
    return {"queries": len(differences), "delta": float(differences.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "familywise_ci": np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist(),
            "family_tests": family_tests, "resamples": repetitions, "seed": SEED,
            "wins": int((differences > 1e-12).sum()),
            "ties": int((abs(differences) <= 1e-12).sum()),
            "losses": int((differences < -1e-12).sum())}


def feature_activity(records, rows):
    """Count feature variation inside a query pool; zero variation cannot rank it."""
    import numpy as np
    activity = {name: {"nonzero_queries": 0, "varying_queries": 0,
                       "nonzero_pairs": 0, "queries": len(records), "pairs": len(rows)}
                for name in GROUPS}
    offset = 0
    for record in records:
        count = len(record["candidate_ids"])
        pool = rows[offset:offset + count]
        for name, columns in GROUPS.items():
            values = pool[:, columns]
            activity[name]["nonzero_queries"] += int(np.any(values != 0))
            activity[name]["varying_queries"] += int(np.any(np.ptp(values, axis=0) > 1e-12))
            activity[name]["nonzero_pairs"] += int(np.any(values != 0, axis=1).sum())
        offset += count
    return activity


def render_report(report):
    lines = ["# Neural ranker ablation (retrospective)", "",
        "Same 500 training / 200 validation / 200 previously evaluated test queries; all judged candidates per query.",
        "Every learned variant is retrained with the same 7 leaves, 200 rounds and training labels. No new model is selected.", "",
        "| Variant | Features | Test NDCG@10 | P@10 | Exact@1 |",
        "|---|---:|---:|---:|---:|"]
    for name, result in report["methods"].items():
        score = result["test"]
        lines.append(f"| {name} | {result['feature_count']} | {score['ndcg10']:.6f} | {score['precision10']:.6f} | {score['exact_at1']:.6f} |")
    lines += ["", "Delta is left minus right. Positive means the left variant performed better.", "",
              "| Contrast | Delta NDCG@10 | Paired 95% CI | Family-corrected CI |",
              "|---|---:|---|---|"]
    for name, contrast in report["contrasts"].items():
        s = contrast["metrics"]["ndcg10"]
        ci = lambda x: f"[{x[0]:+.6f}, {x[1]:+.6f}]"
        lines.append(f"| {name}: {contrast['left']} − {contrast['right']} | {s['delta']:+.6f} | {ci(s['ci95'])} | {ci(s['familywise_ci'])} |")
    lines += ["", "## Interpretation limits", ""]
    lines += ["- " + limitation for limitation in report["limitations"]]
    return "\n".join(lines) + "\n"


def run(args):
    import numpy as np
    out = Path(args.output)
    protocol = protocol_binding(args)
    if args.phase == "freeze":
        out.mkdir(parents=True, exist_ok=True)
        if (out / "protocol.json").exists():
            raise FileExistsError("protocol already frozen; choose a new output directory")
        save_json(out / "protocol.json", {"frozen_at": datetime.now(timezone.utc).isoformat(), **protocol})
        print(json.dumps({"phase": "frozen", "protocol_sha256": file_sha256(out / "protocol.json")}))
        return
    frozen = read_json(out / "protocol.json")
    if {k: v for k, v in frozen.items() if k != "frozen_at"} != json.loads(json.dumps(protocol)):
        raise ValueError("implementation, inputs or protocol changed after freeze")
    if (out / "report.json").exists():
        raise FileExistsError("diagnostic already completed; reports are immutable")
    lgb = load_lightgbm(args.lightgbm_site)
    if lgb.__version__ != protocol["lightgbm_version"]:
        raise ValueError("LightGBM version must match the frozen reference")
    started = time.perf_counter()
    data, reference = Path(args.data), Path(args.reference)
    catalog = load_jsonl(data / "catalog.jsonl")
    # Training may inspect train/validation only. Test files were byte-hashed at
    # freeze, but their records/features are not opened until all models exist.
    splits = {s: load_jsonl(data / f"{s}.jsonl") for s in ("train", "val")}
    validate_dataset(catalog, splits)
    splits["train"] = sorted(splits["train"], key=lambda r: fingerprint([SEED, "accuracy_train", r["query_id"]]))[:500]
    if [r["query_id"] for r in splits["train"]] != read_json(reference / "protocol.json")["train_query_ids"]:
        raise ValueError("training query subset/order differs from frozen reference")
    if [len(splits[s]) for s in ("train", "val")] != [500, 200]:
        raise ValueError("expected historical 500/200 query development protocol")
    extractor = CompactFeatureExtractor(catalog)
    frozen_model = read_json(reference / "model.json")
    ce = LocalCrossEncoder(args.ce_model, cache_dir=out / "ce-cache", threads=2, max_length=128,
                           batch_size=32, precision="dynamic_int8")
    if ce.fingerprint != protocol["encoder_fingerprint"]:
        raise ValueError("encoder/runtime fingerprint mismatch")
    x = {split: verified_features(records, extractor, ce, out, split, args.feature_cache)
         for split, records in splits.items()}
    labels = np.asarray([r["qrels"][p] for r in splits["train"] for p in r["candidate_ids"]])
    groups = [len(r["candidate_ids"]) for r in splits["train"]]
    # Fit every prespecified variant before looking at any new test contrast.
    models, train_times = {}, {}
    for name, columns in VARIANTS.items():
        begin = time.perf_counter()
        ds = lgb.Dataset(x["train"][:, columns], label=labels, group=groups,
                         feature_name=[ACCURACY_FEATURES[i] for i in columns], free_raw_data=False)
        models[name] = lgb.train(PARAMS, ds, num_boost_round=ITERATIONS)
        models[name].save_model(str(out / (name + ".txt")))
        train_times[name] = time.perf_counter() - begin
        print(json.dumps({"phase": "fit", "variant": name, "features": len(columns)}), flush=True)
    save_json(out / "fit_lock.json", {"protocol_sha256": file_sha256(out / "protocol.json"),
        "before_test_read": True, "model_sha256": {name: file_sha256(out / (name + ".txt")) for name in models}})
    splits["test"] = load_jsonl(data / "test.jsonl")
    validate_dataset(catalog, splits)
    if len(splits["test"]) != 200:
        raise ValueError("expected historical 200-query test protocol")
    x["test"] = verified_features(splits["test"], extractor, ce, out, "test", args.feature_cache)
    verification = {}
    for split in ("val", "test"):
        expected = np.asarray(model_scores(frozen_model, x[split]))
        actual = models["full15"].predict(x[split])
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise AssertionError("full retraining does not reproduce frozen predictions")
        verification[split + "_max_abs_prediction_error"] = error
    results, local_rows = {}, {}
    for name, columns in {"bm25": [0], "cross_encoder": [12], **VARIANTS}.items():
        result = {"feature_count": len(columns), "feature_names": [ACCURACY_FEATURES[i] for i in columns],
                  "trained_on_esci": name in models, "train_seconds": train_times.get(name, 0.)}
        for split in ("val", "test"):
            scores = models[name].predict(x[split][:, columns]) if name in models else x[split][:, columns[0]]
            measured = metrics(splits[split], scores)
            result[split] = measured["summary"]
            local_rows[(name, split)] = measured["per_query"]
        results[name] = result
    original = read_json(reference / "report.json")["methods"]["selected"]
    if results["full15"]["test"] != original:
        raise AssertionError("full model metrics differ from historical published result")
    contrasts = {name: {"left": left, "right": right, "metrics": {
        key: paired_contrast(local_rows[(left, "test")], local_rows[(right, "test")], key,
                             family_tests=len(CONTRASTS) if key == "ndcg10" else 1)
        for key in METRICS}} for name, (left, right) in CONTRASTS.items()}
    report = {"status": "completed", "scope": "retrospective_fixed_pool_neural_ablation",
        "test_previously_seen": True, "new_model_selection": False,
        "protocol_sha256": file_sha256(out / "protocol.json"),
        "source_files": protocol["source_files"],
        "data": {"catalog_products": len(catalog), "queries": {k: len(v) for k, v in splits.items()},
                 "candidate_pairs": {k: len(v) for k, v in x.items()}, "files_sha256": protocol["data_files"]},
        "runtime": {"numpy": np.__version__, "lightgbm": lgb.__version__, "encoder": ce.config},
        "verification": verification, "methods": results, "contrasts": contrasts,
        "feature_activity": {s: feature_activity(splits[s], x[s]) for s in ("train", "val", "test")},
        "model_sha256": {name: file_sha256(out / (name + ".txt")) for name in models},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "The historical test labels/results were already observed. This is retrospective diagnosis, not a new untouched-test gain claim.",
            "Fixed hyperparameters estimate removal effects under this training recipe; they do not find the best individually tuned alternative.",
            "Feature interactions and correlated substitutes mean these removal effects are not additive percentages or unique causal attribution.",
            "without_ce_all is lexical12; without_pool_stats is lexical+raw-CE13. without_ce_raw still retains two CE-derived statistics and is NOT a no-neural baseline.",
            "NDCG contrasts include Bonferroni-corrected paired bootstrap intervals; secondary-metric 95% intervals are exploratory and unadjusted.",
            "Bootstrap resamples queries, not training runs. It does not quantify training-split uncertainty or upstream pretraining overlap.",
            "An inactive group cannot demonstrate benefit on these queries. Nonzero/within-pool-varying feature counts are included for every split.",
            "All candidates in each pool are judged. This result does not measure all-catalog retrieval, new images, personalization, online CTR or business lift.",
            "Frozen historical inputs and feature caches are local licensed assets, excluded from Git. Checksums verify identity; source checkout alone does not reconstruct the historical subset."
        ]}
    if json.loads(json.dumps(protocol_binding(args))) != {k: v for k, v in frozen.items() if k != "frozen_at"}:
        raise ValueError("inputs or implementation changed while the experiment ran")
    save_json(out / "per_query_local.json", {name + "/" + split: rows for (name, split), rows in local_rows.items()})
    save_json(out / "report.json", report)
    (out / "report.md").write_text(render_report(report), encoding="utf-8")
    ce.close()
    print(json.dumps({"phase": "completed", "full15": results["full15"]["test"], "seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("--data", required=True, help="matching frozen esci-v2 directory")
    parser.add_argument("--reference", required=True, help="original accuracy-v1 directory")
    parser.add_argument("--ce-model", required=True, help="prepared fixed official local encoder")
    parser.add_argument("--feature-cache", help="optional original hash-bound feature arrays; otherwise recompute")
    parser.add_argument("--output", default=str(ROOT / "outputs/ablation-neural-v1"))
    parser.add_argument("--lightgbm-site", help="optional explicit directory containing the matching LightGBM installation")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
