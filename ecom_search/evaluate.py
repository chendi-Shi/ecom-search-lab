"""Graded fixed-pool evaluation and one validation-only model selection."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from .data import file_sha256, load_dataset, save_json, save_jsonl


def _validate_metric_input(ranked_ids, qrels, k):
    if type(k) is not int or k < 1:
        raise ValueError("Metric cutoff must be a positive integer")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Metric rankings cannot contain duplicate product ids")
    if any(type(grade) is not int or not 0 <= grade <= 3 for grade in qrels.values()):
        raise ValueError("Relevance grades must be integers 0..3")


def ndcg(ranked_ids, qrels, k=10):
    _validate_metric_input(ranked_ids, qrels, k)
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    if not idcg:
        return None
    dcg = sum((2 ** qrels.get(pid, 0) - 1) / math.log2(rank + 2)
              for rank, pid in enumerate(ranked_ids[:k]))
    return dcg / idcg


def recall(ranked_ids, qrels, k=10):
    _validate_metric_input(ranked_ids, qrels, k)
    relevant = {pid for pid, grade in qrels.items() if grade >= 2}
    if not relevant:
        return None
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def _aggregate(rows, k):
    metrics = {"queries": len(rows)}
    for name in (f"ndcg@{k}", f"candidate_recall@{k}"):
        values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
        metrics[name] = sum(values) / len(values) if values else None
        metrics[f"{name}_denominator"] = len(values)
    return metrics


def _format_metric(value, signed=False):
    return "null" if value is None else format(value, "+.6f" if signed else ".6f")


def evaluate_ranking(records, rank_fn, k=10):
    """rank_fn(record) must return every fixed-pool candidate exactly once."""
    if type(k) is not int or k < 1:
        raise ValueError("Metric cutoff must be a positive integer")
    rows = []
    for record in records:
        ranking = list(rank_fn(record))
        candidates = record["candidate_ids"]
        if len(ranking) != len(candidates) or len(set(ranking)) != len(ranking) or set(ranking) != set(candidates):
            raise ValueError("Ranker must return each fixed-pool candidate exactly once")
        if set(record["qrels"]) != set(candidates):
            raise ValueError("Every fixed-pool candidate requires an independent judgment")
        query_tokens = re.findall(r"\w+", record["query"].casefold())
        rows.append({"query_id": record["query_id"], "query": record["query"],
                     "long_query": len(query_tokens) >= 5, "query_token_count": len(query_tokens),
                     "candidate_count": len(candidates), "ranking": ranking,
                     "metrics": {f"ndcg@{k}": ndcg(ranking, record["qrels"], k),
                                 f"candidate_recall@{k}": recall(ranking, record["qrels"], k)}})
    return {"metrics": _aggregate(rows, k), "slices": {
                "long_query_ge_5_tokens": _aggregate([row for row in rows if row["long_query"]], k),
                "short_query_lt_5_tokens": _aggregate([row for row in rows if not row["long_query"]], k)},
            "bad_cases": [row for row in rows if row["metrics"][f"ndcg@{k}"] is not None
                          and row["metrics"][f"ndcg@{k}"] < 0.8],
            "per_query": rows}


def _scored_rank(record, catalog, scorer):
    scores = [(pid, float(scorer(record["query"], catalog[pid]))) for pid in record["candidate_ids"]]
    if any(not math.isfinite(score) for _, score in scores):
        raise ValueError("Ranker must produce finite candidate scores")
    return [pid for pid, _ in sorted(scores, key=lambda item: (-item[1], item[0]))]


def train_and_evaluate(data_dir, output_dir):
    from .ranking import FeatureExtractor, PairwiseRanker
    from .index import catalog_digest

    started = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_files = ("model.json", "report.json", "report.md", "per_query_test.jsonl")
    if any((output_dir / name).exists() for name in output_files):
        raise FileExistsError("An evaluation artifact already exists; choose a new output directory")
    catalog, train, val, test = load_dataset(data_dir)
    if not train or not val or not test:
        raise ValueError("Training, validation and final test splits must all be nonempty")
    fingerprints = {name: file_sha256(data_dir / name) for name in (
        "catalog.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "manifest.json") if (data_dir / name).exists()}
    extractor = FeatureExtractor(catalog)
    catalog_by_id = {product["id"]: product for product in catalog}
    baseline_rank = lambda record: _scored_rank(record, catalog_by_id, extractor.bm25)
    baseline_validation = evaluate_ranking(val, baseline_rank)
    selection, models = [], {}
    for epochs in (0, 5, 15, 30):
        model = PairwiseRanker()
        training_start = time.perf_counter()
        info = model.fit(train, extractor, epochs=epochs)
        fit_seconds = time.perf_counter() - training_start
        rank_fn = lambda record, fitted=model: _scored_rank(
            record, catalog_by_id, lambda query, product: fitted.score(extractor.features(query, product)))
        validation = evaluate_ranking(val, rank_fn)
        selection.append({"epochs": epochs, "validation_metrics": validation["metrics"],
                          "fit_seconds": fit_seconds, "training_info": info})
        models[epochs] = model
    criterion = "Highest validation macro NDCG@10; ties select fewer epochs; fixed epoch grid [0,5,15,30]"
    eligible = [row for row in selection if row["validation_metrics"]["ndcg@10"] is not None]
    if not eligible:
        raise ValueError("Validation split has no nonzero ideal DCG")
    chosen = max(eligible, key=lambda row: (row["validation_metrics"]["ndcg@10"], -row["epochs"]))
    selected = models[chosen["epochs"]]
    for name, fingerprint in fingerprints.items():
        if file_sha256(data_dir / name) != fingerprint:
            raise ValueError("Dataset changed during validation-only selection")
    # The selected model and data binding are saved before any final test scoring.
    frozen_model = {"ranker": selected.to_dict(), "catalog_sha256": catalog_digest(catalog),
                    "selected_epochs": chosen["epochs"],
                    "selection_criterion": criterion, "data_fingerprints": fingerprints,
                    "frozen_before_test": True, "feature_names": list(extractor.names)}
    save_json(output_dir / "model.json", frozen_model)
    restored = PairwiseRanker.from_dict(frozen_model["ranker"])
    learned_rank = lambda record: _scored_rank(
        record, catalog_by_id, lambda query, product: restored.score(extractor.features(query, product)))
    test_start = time.perf_counter()
    baseline_test = evaluate_ranking(test, baseline_rank)
    learned_test = evaluate_ranking(test, learned_rank)
    test_seconds = time.perf_counter() - test_start
    test_rows = [{"query_id": left["query_id"], "query": left["query"], "long_query": left["long_query"],
                  "candidate_ids": record["candidate_ids"], "qrels": record["qrels"],
                  "title_bm25": left, "learned_pairwise": right,
                  "ndcg_delta": right["metrics"]["ndcg@10"] - left["metrics"]["ndcg@10"]
                      if right["metrics"]["ndcg@10"] is not None and left["metrics"]["ndcg@10"] is not None else None}
                 for record, left, right in zip(test, baseline_test["per_query"], learned_test["per_query"])]
    deltas = {name: learned_test["metrics"][name] - baseline_test["metrics"][name]
              if learned_test["metrics"][name] is not None and baseline_test["metrics"][name] is not None else None
              for name in ("ndcg@10", "candidate_recall@10")}
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Same fixed 20 fully human-judged ESCI candidates/query; candidate recall is not corpus retrieval recall",
        "gain": "2**grade - 1", "relevant_grade_min": 2, "ranking_cutoff": 10,
        "split_queries": {"train": len(train), "val": len(val), "test": len(test)},
        "catalog_products": len(catalog), "data_fingerprints": fingerprints,
        "selection_criterion": criterion, "validation_selection": selection,
        "selected_epochs": chosen["epochs"], "model_weights": frozen_model["ranker"],
        "title_bm25_validation": baseline_validation["metrics"],
        "test": {"title_bm25": {key: value for key, value in baseline_test.items() if key != "per_query"},
                 "learned_pairwise": {key: value for key, value in learned_test.items() if key != "per_query"}},
        "test_delta_learned_minus_baseline": deltas,
        "timing_seconds": {"final_test_scoring": test_seconds, "total": time.perf_counter() - started},
        "protocol": "All epoch decisions used train/val only; one final evaluation of the selected model and baseline; no test tuning",
        "limits": ["Small public US text-only subset; no personalized, multimodal or industrial-scale claim",
                   "Long-query slice uses >=5 word tokens, not human-annotated semantic complexity",
                   "Offline descriptive metric deltas do not establish statistical significance or online business impact"],
    }
    save_jsonl(output_dir / "per_query_test.jsonl", test_rows)
    save_json(output_dir / "report.json", report)
    lines = ["# 电商候选排序实验", "", report["scope"], "",
             f"训练/验证/测试 Query：{len(train)}/{len(val)}/{len(test)}；商品池：{len(catalog):,}。",
             "", "| 模型 | Test NDCG@10 | Test Candidate Recall@10 |", "|---|---:|---:|"]
    for name in ("title_bm25", "learned_pairwise"):
        metrics = report["test"][name]["metrics"]
        lines.append(f"| {name} | {_format_metric(metrics['ndcg@10'])} | {_format_metric(metrics['candidate_recall@10'])} |")
    lines += ["", f"验证选择：{criterion}；最终 epochs={chosen['epochs']}。",
              f"Test NDCG 差值（学习模型减基线）：{_format_metric(deltas['ndcg@10'], signed=True)}。",
              "", "分级收益为 `2**grade-1`，Recall 相关条件为 grade≥2；无相关商品的 Query 排除于相应宏平均，分母保留在 JSON。",
              "", "模型和数据指纹在测试评分前冻结。完整参数选择、权重、耗时、长 Query 切片和差例见 report.json；逐 Query 对照见 per_query_test.jsonl。",
              "", "这是固定候选内排序；文本特征线性模型不能代表深度、多模态、个性化模型，离线差值不代表线上收益。"]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
