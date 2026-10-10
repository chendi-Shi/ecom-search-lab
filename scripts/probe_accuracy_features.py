"""Validation-only, predeclared cheap feature probe; never opens holdout files.

Reuses SHA-bound train/val neural scores. No model, selection, or caches are
modified. Additional query-relative features require a whole candidate group.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

from ecom_search.data import file_sha256, load_jsonl, save_json
from ecom_search.evaluate import ndcg


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metrics(records, scores):
    offset, ndcgs, precisions, exact = 0, [], [], []
    for record in records:
        ids = record["candidate_ids"]
        values = scores[offset:offset + len(ids)]
        if len(values) != len(ids):
            raise ValueError("candidate/score count mismatch")
        ranked = [pid for _, pid in sorted(zip(values, ids), key=lambda pair: (-float(pair[0]), pair[1]))]
        value = ndcg(ranked, record["qrels"], 10)
        if value is not None:
            ndcgs.append(value)
        precisions.append(sum(record["qrels"][pid] >= 2 for pid in ranked[:10]) / min(10, len(ids)))
        exact.append(float(record["qrels"][ranked[0]] == 3))
        offset += len(ids)
    if offset != len(scores):
        raise ValueError("extra scores")
    return {"queries": len(records), "ndcg10": sum(ndcgs) / len(ndcgs),
        "ndcg10_denominator": len(ndcgs), "precision10": sum(precisions) / len(precisions),
        "exact_at1": sum(exact) / len(exact)}


def augment(records, matrix, old_weights):
    import numpy as np
    from scipy.stats import rankdata
    relatives, offset = [], 0
    for record in records:
        count = len(record["candidate_ids"])
        logits = matrix[offset:offset + count, -1]
        percentile = (rankdata(logits, method="average") - 1) / max(1, count - 1)
        deviation = float(logits.std())
        zscore = (logits - logits.mean()) / deviation if deviation > 1e-12 else np.zeros(count)
        relatives.extend(zip(percentile, zscore, [float(len(record["query"].split()))] * count))
        offset += count
    if offset != len(matrix):
        raise ValueError("incorrect grouped feature rows")
    relatives = np.asarray(relatives, dtype=np.float64)
    family_a = np.column_stack((matrix, relatives[:, :2]))
    family_b = np.column_stack((family_a, matrix[:, :12] @ old_weights, relatives[:, 2]))
    return {"A": family_a, "B": family_b}


def run(args):
    started = time.perf_counter()
    if args.lightgbm_site:
        sys.path.append(str(Path(args.lightgbm_site).resolve()))
    import numpy as np
    import lightgbm as lgb
    out, data = ROOT / "outputs/accuracy-v1", ROOT / "data/esci-v2"
    # Fail closed if later edits accidentally try to read a v2 holdout or result.
    def prohibit_holdout(event, event_args):
        if event != "open" or not isinstance(event_args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(event_args[0])).resolve()
        if path.parent == data and "test" in path.name.lower():
            raise PermissionError("validation-only probe cannot read v2 holdout")
        if path.parent == out and ("test" in path.name.lower() or path.name in
                ("final-lock.json", "report.json", "end_to_end_report.json")):
            raise PermissionError("validation-only probe cannot read final artifacts")
    sys.addaudithook(prohibit_holdout)

    development = out / "development-initial"
    source = development if (development / "protocol.json").exists() else out
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if len(protocol["features"]) != 13:
        raise ValueError("probe requires the archived original 13-feature development cache")
    all_train = load_jsonl(data / "train.jsonl")
    by_id = {record["query_id"]: record for record in all_train}
    if len(by_id) != len(all_train) or len(set(protocol["train_query_ids"])) != len(protocol["train_query_ids"]):
        raise ValueError("duplicate train query IDs")
    records = {"train": [by_id[qid] for qid in protocol["train_query_ids"]],
               "val": load_jsonl(data / "val.jsonl")}
    if {r["query_id"] for r in records["train"]} & {r["query_id"] for r in records["val"]}:
        raise ValueError("train/val query overlap")
    verified, matrices = {}, {}
    for split, rows in records.items():
        metadata = json.loads((source / f"features-{split}.json").read_text(encoding="utf-8"))
        path = source / f"features-{split}.npy"
        actual_hash = file_sha256(path)
        if metadata["sha256"] != actual_hash or fingerprint(metadata["binding"]) != metadata["key"]:
            raise ValueError("feature file or metadata integrity mismatch")
        if metadata["binding"]["records_sha256"] != fingerprint(rows):
            raise ValueError("feature record order mismatch")
        if metadata["binding"]["features"] != protocol["features"]:
            raise ValueError("feature name mismatch")
        matrix = np.load(path, allow_pickle=False)
        expected = (sum(len(r["candidate_ids"]) for r in rows), len(protocol["features"]))
        if matrix.shape != expected or not np.isfinite(matrix).all():
            raise ValueError("feature shape/finite check failed")
        for row in rows:
            if set(row["candidate_ids"]) != set(row["qrels"]) or len(set(row["candidate_ids"])) != len(row["candidate_ids"]):
                raise ValueError("candidate judgments inconsistent")
        matrices[split] = matrix
        verified[split] = {"npy_sha256": actual_hash, "shape": list(matrix.shape),
            "queries": len(rows), "records_sha256": fingerprint(rows), "metadata_key": metadata["key"]}

    old_path = ROOT / "outputs/ranking/model.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    weights = np.asarray(old["ranker"]["weights"], dtype=np.float64)
    if old["feature_names"] != protocol["features"][:12] or weights.shape != (12,):
        raise ValueError("old pairwise feature schema mismatch")
    base = json.loads((source / "validation.json").read_text(encoding="utf-8"))["methods"]
    computed_old = metrics(records["val"], matrices["val"][:, :12] @ weights)
    computed_ce = metrics(records["val"], matrices["val"][:, -1])
    for name, computed in (("transferred_pairwise", computed_old), ("cross_encoder", computed_ce)):
        for metric in ("ndcg10", "precision10", "exact_at1"):
            if abs(computed[metric] - base[name][metric]) > 1e-12:
                raise ValueError("baseline reproduction mismatch")
    expanded = {split: augment(rows, matrices[split], weights) for split, rows in records.items()}
    names = {"A": protocol["features"] + ["ce_query_percentile", "ce_query_zscore"],
             "B": protocol["features"] + ["ce_query_percentile", "ce_query_zscore", "old_pairwise_score", "query_token_count"]}
    labels = np.asarray([r["qrels"][pid] for r in records["train"] for pid in r["candidate_ids"]])
    groups = [len(r["candidate_ids"]) for r in records["train"]]
    grid, best = [], {}
    for family in ("A", "B"):
        for leaves in (7, 15):
            params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "label_gain": [0, 1, 3, 7],
                "learning_rate": .05, "num_leaves": leaves, "min_data_in_leaf": 40,
                "lambda_l2": 1, "num_threads": 2, "deterministic": True, "force_col_wise": True,
                "verbosity": -1, "seed": 20261010, "feature_pre_filter": False}
            dataset = lgb.Dataset(expanded["train"][family], label=labels, group=groups,
                feature_name=names[family], free_raw_data=False)
            booster = lgb.train(params, dataset, num_boost_round=400)
            for rounds in (50, 100, 200, 400):
                score = metrics(records["val"], booster.predict(expanded["val"][family], num_iteration=rounds))
                entry = {"family": family, "leaves": leaves, "iterations": rounds, **score}
                grid.append(entry)
                print(json.dumps(entry), flush=True)
                if family not in best or score["ndcg10"] > best[family]["ndcg10"] + 1e-12:
                    best[family] = {**entry, "params": params, "feature_names": names[family],
                        "feature_importance_gain": dict(zip(names[family], booster.feature_importance("gain", iteration=rounds).tolist())),
                        "feature_importance_split": dict(zip(names[family], booster.feature_importance("split", iteration=rounds).tolist()))}
    chosen = max(best.values(), key=lambda item: item["ndcg10"])
    delta = chosen["ndcg10"] - base["selected"]["ndcg10"]
    report = {"status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started, "lightgbm_version": lgb.__version__,
        "scope": "Only train/val; no holdout read or scoring; no neural encoding; no model/cache/selection mutation.",
        "inputs": verified, "old_model_sha256": file_sha256(old_path),
        "script_sha256": file_sha256(Path(__file__)), "baseline_validation": base,
        "feature_definitions": {"A": "13 original features + within-query tied-average percentile of CE logit in [0,1] + CE logit population z-score (zero for constant groups)",
            "B": "A + original 12 features dot frozen old pairwise weights + len(query.split())",
            "candidate_dependency": "Relative features depend on the complete candidate group; production pools may have different distributions."},
        "grid": grid, "best_per_family": best, "best_spec": {k: chosen[k] for k in ("family", "leaves", "iterations")},
        "ndcg10_delta_vs_selected_13_features": delta, "minimum_complexity_justification_delta": .005,
        "recommendation": "Consider validation winner before fresh holdout freeze." if delta >= .005 else
            "Do not adopt extra candidate-dependent features: validation improvement below prespecified 0.005 threshold.",
        "limitations": ["Exploratory validation-only result after an earlier grid, not a final generalization claim.",
            "No new holdout result was accessed; only a fresh frozen evaluation can establish improvement."]}
    destination = out / "validation_feature_probe_reproduction.json"
    if destination.exists():
        raise ValueError("refusing to overwrite a previous probe reproduction")
    save_json(destination, report)
    print(json.dumps({"best_spec": report["best_spec"], "delta": delta, "recommendation": report["recommendation"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm-site")
    run(parser.parse_args())
